#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <vector>

#include <time.h>
#include <unistd.h>

#ifdef USE_NVML_TELEMETRY
#include <nvml.h>
#else
typedef void* nvmlDevice_t;
#endif

// GPU energy is measured via two point-in-time reads of NVML's monotonic
// energy counter (nvmlDeviceGetTotalEnergyConsumption) at segment begin/end,
// not via periodic power sampling: a background sampler needs at least two
// wakeups inside the measured window to integrate anything, and at these
// grid sizes the window (0.2-2.1 ms) is routinely shorter than a single
// sampling interval, so the old thread-based sampler produced NaN in the
// large majority of runs. A counter delta needs no samples "inside" the
// window -- one read before, one read after, subtract -- so it is exact
// regardless of how short the window is. See Fase_2/telemetry.cuh
// (EnergyProbe) for the same technique applied to a different CSV schema.
struct PowerBuffer {
    nvmlDevice_t device;
    bool nvml_enabled;
    bool segment_active;       // true between start_sampling and stop_sampling
    bool capture_failed;       // a read failed or the monotonicity guard tripped since the last clear
    bool capture_has_data;     // at least one begin/end delta was accumulated since the last clear
    unsigned long long segment_begin_mj;  // energy counter value latched by start_sampling
    unsigned long long segment_begin_ns;  // wall-clock instant latched by start_sampling
    double accumulated_j;      // running total of closed-segment energy deltas since the last clear
    double accumulated_window_s;  // running total of closed-segment wall-clock durations since the last clear
};

struct RAEnergySnapshot {
    double energy_j;
    unsigned long long timestamp_ns;
    bool valid;
};

struct EnergyMeasurement {
    bool gpu_valid = false;
    bool cpu_valid = false;
    double time_total_s = 0.0;
    double energy_gpu_j = 0.0;
    double energy_cpu_j = 0.0;
    double energy_total_j = 0.0;
    double edp_j_s = 0.0;
    double joules_per_gflop = 0.0;
    double avg_power_w = 0.0;
    // Aliases kept for the pre-existing optional --csv output.
    double energy_j = 0.0;
    double edp = 0.0;
};

static std::string energy_field(bool valid, double value) {
    if (!valid || !std::isfinite(value)) return "NA";
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.6e", value);
    return buffer;
}

static unsigned long long power_sampling_now_ns() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return static_cast<unsigned long long>(ts.tv_sec) * 1000000000ULL +
           static_cast<unsigned long long>(ts.tv_nsec);
}

#ifdef USE_NVML_TELEMETRY

static bool& telemetry_nvml_initialized_flag() {
    static bool value = false;
    return value;
}

static bool& telemetry_nvml_enabled_flag() {
    static bool value = false;
    return value;
}

static nvmlDevice_t& telemetry_nvml_device_ref() {
    static nvmlDevice_t value = nullptr;
    return value;
}

static bool telemetry_nvml_initialize(int device_id) {
    if (telemetry_nvml_initialized_flag()) {
        return telemetry_nvml_enabled_flag();
    }
    telemetry_nvml_initialized_flag() = true;

    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::fprintf(stderr,
                     "ADVERTENCIA: nvmlInit fallo: %s. EDP no sera medida. "
                     "Continuando sin telemetria GPU.\n",
                     nvmlErrorString(result));
        return false;
    }

    result = nvmlDeviceGetHandleByIndex(static_cast<unsigned int>(device_id),
                                        &telemetry_nvml_device_ref());
    if (result != NVML_SUCCESS) {
        std::fprintf(stderr,
                     "ADVERTENCIA: no se pudo obtener el dispositivo NVML: %s. "
                     "Continuando sin telemetria GPU.\n",
                     nvmlErrorString(result));
        return false;
    }

    telemetry_nvml_enabled_flag() = true;
    unsigned int power_limit = 0;
    result = nvmlDeviceGetPowerManagementLimit(telemetry_nvml_device_ref(), &power_limit);
    if (result == NVML_SUCCESS) {
        std::printf("NVML inicializado. Power Limit: %.2f W\n", power_limit / 1000.0);
    } else {
        std::printf("NVML inicializado. Power Limit: NaN W\n");
    }
    return true;
}

static bool telemetry_nvml_enabled() {
    return telemetry_nvml_enabled_flag();
}

static nvmlDevice_t telemetry_nvml_device() {
    return telemetry_nvml_device_ref();
}

#else

static bool telemetry_nvml_initialize(int) {
    static bool warned = false;
    if (!warned) {
        std::fprintf(stderr,
                     "ADVERTENCIA: NVML no esta disponible en tiempo de compilacion. "
                     "EDP GPU no sera medida. Continuando sin telemetria GPU.\n");
        warned = true;
    }
    return false;
}

#endif

static PowerBuffer* power_buffer_create(int device_id) {
    PowerBuffer* pb = new PowerBuffer();
    pb->device = nullptr;
    pb->nvml_enabled = false;
    pb->segment_active = false;
    pb->capture_failed = false;
    pb->capture_has_data = false;
    pb->segment_begin_mj = 0;
    pb->segment_begin_ns = 0;
    pb->accumulated_j = 0.0;
    pb->accumulated_window_s = 0.0;
#ifdef USE_NVML_TELEMETRY
    if (telemetry_nvml_enabled() && device_id == 0) {
        pb->device = telemetry_nvml_device();
        pb->nvml_enabled = (pb->device != nullptr);
    } else if (telemetry_nvml_enabled() &&
               nvmlDeviceGetHandleByIndex(static_cast<unsigned int>(device_id), &pb->device) ==
                   NVML_SUCCESS) {
        pb->nvml_enabled = true;
    }
#else
    (void)device_id;
#endif
    return pb;
}

// Reinicia el acumulado corriente (energia y ventana) y cualquier estado de
// invalidez previo. Se llama tras el warm-up y entre tramos de energia (ver
// close_energy_segment en stencil_tensor_activation.cu) para que cada tramo
// arranque desde cero.
static void power_buffer_samples_clear(PowerBuffer* pb) {
    if (pb == nullptr) return;
    pb->accumulated_j = 0.0;
    pb->accumulated_window_s = 0.0;
    pb->capture_failed = false;
    pb->capture_has_data = false;
}

// Abre un segmento medido: lee el contador de energia acumulada de NVML y el
// reloj de pared una sola vez cada uno, y los guarda como marca de inicio.
// No hay hilo que arrancar; el segmento se cierra con una segunda lectura
// puntual de ambos en power_buffer_stop_sampling.
static void power_buffer_start_sampling(PowerBuffer* pb) {
    if (pb == nullptr || !pb->nvml_enabled) return;
    if (pb->segment_active) return;
    pb->segment_active = true;
    pb->segment_begin_ns = power_sampling_now_ns();
#ifdef USE_NVML_TELEMETRY
    unsigned long long energy_mj = 0;
    if (nvmlDeviceGetTotalEnergyConsumption(pb->device, &energy_mj) == NVML_SUCCESS) {
        pb->segment_begin_mj = energy_mj;
    } else {
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr,
                         "ADVERTENCIA: nvmlDeviceGetTotalEnergyConsumption fallo al abrir "
                         "el segmento; energia GPU sera NaN.\n");
            warned = true;
        }
        pb->capture_failed = true;
    }
#endif
}

// Cierra el segmento medido: lee el contador y el reloj de pared una segunda
// vez, calcula los deltas contra las marcas de inicio de ESTE segmento y los
// acumula. Guarda de monotonia sobre el contador de energia: si la lectura
// final es menor que la inicial, el segmento se marca invalido en vez de
// acumular un delta negativo. La duracion de la ventana se acumula siempre
// (es un reloj de pared real, no depende de NVML) pero
// power_buffer_window_seconds solo la expone cuando la captura es valida,
// igual que power_buffer_energy_joules.
static void power_buffer_stop_sampling(PowerBuffer* pb) {
    if (pb == nullptr || !pb->nvml_enabled || !pb->segment_active) return;
    pb->segment_active = false;
    const unsigned long long end_ns = power_sampling_now_ns();
    if (end_ns > pb->segment_begin_ns) {
        pb->accumulated_window_s +=
            static_cast<double>(end_ns - pb->segment_begin_ns) / 1e9;
    }
#ifdef USE_NVML_TELEMETRY
    unsigned long long energy_mj = 0;
    if (nvmlDeviceGetTotalEnergyConsumption(pb->device, &energy_mj) != NVML_SUCCESS) {
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr,
                         "ADVERTENCIA: nvmlDeviceGetTotalEnergyConsumption fallo al cerrar "
                         "el segmento; energia GPU sera NaN.\n");
            warned = true;
        }
        pb->capture_failed = true;
        return;
    }
    if (energy_mj < pb->segment_begin_mj) {
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr,
                         "ADVERTENCIA: el contador de energia NVML no fue monotono "
                         "dentro del segmento; energia GPU sera NaN para ese tramo.\n");
            warned = true;
        }
        pb->capture_failed = true;
        return;
    }
    pb->accumulated_j += static_cast<double>(energy_mj - pb->segment_begin_mj) / 1000.0;
    pb->capture_has_data = true;
#endif
}

static bool power_buffer_capture_valid(const PowerBuffer* pb) {
    return pb != nullptr && pb->nvml_enabled && !pb->capture_failed && pb->capture_has_data;
}

// Duracion de la ventana efectivamente medida, con las mismas marcas
// begin/end que power_buffer_energy_joules: permite que el tiempo usado como
// denominador de avg_power_w/edp_j_s provenga del mismo intervalo que la
// energia, en vez de un reloj de pared aparte.
static double power_buffer_window_seconds(const PowerBuffer* pb) {
    if (!power_buffer_capture_valid(pb)) return 0.0;
    return pb->accumulated_window_s;
}

static double power_buffer_energy_joules(const PowerBuffer* pb) {
    if (!power_buffer_capture_valid(pb)) return 0.0;
    return std::isfinite(pb->accumulated_j) ? pb->accumulated_j : 0.0;
}

static void power_buffer_destroy(PowerBuffer* pb) {
    if (pb == nullptr) return;
    power_buffer_stop_sampling(pb);
    delete pb;
}

static bool read_rapl_u64(const std::string& path, unsigned long long& value) {
    std::FILE* file = std::fopen(path.c_str(), "r");
    if (file == nullptr) return false;
    char buffer[64] = {};
    const size_t n = std::fread(buffer, 1, sizeof(buffer) - 1, file);
    std::fclose(file);
    if (n == 0) return false;
    buffer[n] = '\0';
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(buffer, &end, 10);
    if (end == buffer) return false;
    value = parsed;
    return true;
}

static bool read_rapl_energy_joules(double& energy_j) {
    std::vector<std::string> paths;
    const std::string direct = "/sys/class/powercap/intel-rapl/energy_uj";
    unsigned long long value = 0;
    if (read_rapl_u64(direct, value)) {
        paths.push_back(direct);
    } else {
        DIR* directory = opendir("/sys/class/powercap");
        if (directory != nullptr) {
            struct dirent* entry = nullptr;
            while ((entry = readdir(directory)) != nullptr) {
                const std::string name(entry->d_name);
                if (name.rfind("intel-rapl:", 0) != 0 || name.find(':', 11) != std::string::npos) {
                    continue;
                }
                paths.push_back("/sys/class/powercap/" + name + "/energy_uj");
            }
            closedir(directory);
        }
    }

    unsigned long long total_uj = 0;
    bool any = false;
    for (const std::string& path : paths) {
        if (read_rapl_u64(path, value)) {
            total_uj += value;
            any = true;
        }
    }
    energy_j = static_cast<double>(total_uj) / 1e6;
    return any;
}

static double rapl_energy_joules() {
    double energy_j = 0.0;
    return read_rapl_energy_joules(energy_j) ? energy_j : 0.0;
}

static RAEnergySnapshot rapl_snapshot_now() {
    RAEnergySnapshot snapshot{};
    snapshot.timestamp_ns = power_sampling_now_ns();
    snapshot.valid = read_rapl_energy_joules(snapshot.energy_j);
    static bool had_valid_snapshot = false;
    static bool warned = false;
    if (snapshot.valid) {
        had_valid_snapshot = true;
    } else if (had_valid_snapshot && !warned) {
        std::fprintf(stderr,
                     "ADVERTENCIA: lectura RAPL fallo durante el benchmark; "
                     "energia CPU sera NaN para esa ventana.\n");
        warned = true;
    }
    return snapshot;
}

static double rapl_energy_delta(const RAEnergySnapshot& before,
                                const RAEnergySnapshot& after) {
    if (!before.valid || !after.valid || after.energy_j < before.energy_j) return 0.0;
    return after.energy_j - before.energy_j;
}

static bool rapl_available() {
    double energy_j = 0.0;
    if (!read_rapl_energy_joules(energy_j)) return false;
    return std::isfinite(rapl_energy_joules());
}

static EnergyMeasurement make_energy_measurement(const PowerBuffer* pb,
                                                 const RAEnergySnapshot& rapl_before,
                                                 const RAEnergySnapshot& rapl_after,
                                                 double time_total_s,
                                                 double flops_total) {
    EnergyMeasurement result;
    result.time_total_s = time_total_s;
    result.gpu_valid = power_buffer_capture_valid(pb);
    result.cpu_valid = rapl_before.valid && rapl_after.valid &&
                       rapl_after.energy_j >= rapl_before.energy_j;
    if (result.gpu_valid) {
        result.energy_gpu_j = power_buffer_energy_joules(pb);
        result.avg_power_w = (time_total_s > 0.0) ? result.energy_gpu_j / time_total_s : 0.0;
        result.energy_j = result.energy_gpu_j;
    }
    if (result.cpu_valid) {
        result.energy_cpu_j = rapl_energy_delta(rapl_before, rapl_after);
    }
    if (result.gpu_valid && result.cpu_valid) {
        result.energy_total_j = result.energy_gpu_j + result.energy_cpu_j;
        result.edp_j_s = result.energy_total_j * time_total_s;
        result.joules_per_gflop = (flops_total > 0.0)
            ? result.energy_total_j / (flops_total / 1e9) : 0.0;
    }
    result.edp = result.energy_gpu_j * time_total_s;
    return result;
}
