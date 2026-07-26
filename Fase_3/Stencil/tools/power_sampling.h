#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <pthread.h>
#include <string>
#include <vector>

#include <time.h>
#include <unistd.h>

#ifdef USE_NVML_TELEMETRY
#include <nvml.h>
#else
typedef void* nvmlDevice_t;
#endif

struct PowerSample {
    unsigned long long timestamp_ns;
    unsigned int power_mw;
};

struct PowerBuffer {
    std::vector<PowerSample> samples;
    nvmlDevice_t device;
    bool nvml_enabled;
    pthread_t sampling_thread;
    volatile bool stop_sampling;
    bool sampling_active;
    bool sampling_thread_started;
    bool sampling_failed;
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

static void power_buffer_append_sample(PowerBuffer* pb, unsigned long long timestamp_ns,
                                       unsigned int power_mw) {
    pb->samples.push_back(PowerSample{timestamp_ns, power_mw});
}

static void power_buffer_sample_once(PowerBuffer* pb) {
    const unsigned long long timestamp_ns = power_sampling_now_ns();
    unsigned int power_mw = 0;
#ifdef USE_NVML_TELEMETRY
    if (pb->nvml_enabled &&
        nvmlDeviceGetPowerUsage(pb->device, &power_mw) != NVML_SUCCESS) {
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr,
                         "ADVERTENCIA: nvmlDeviceGetPowerUsage fallo; "
                         "energia GPU sera NaN.\n");
            warned = true;
        }
        pb->sampling_failed = true;
        power_mw = 0;
    }
#else
    (void)pb;
#endif
    power_buffer_append_sample(pb, timestamp_ns, power_mw);
}

static void* power_buffer_sampling_main(void* opaque) {
    PowerBuffer* pb = static_cast<PowerBuffer*>(opaque);
    const struct timespec interval = {0, 10000000L};
    while (!pb->stop_sampling) {
        power_buffer_sample_once(pb);
        nanosleep(&interval, nullptr);
    }
    return nullptr;
}

static PowerBuffer* power_buffer_create(int device_id) {
    PowerBuffer* pb = new PowerBuffer();
    pb->device = nullptr;
    pb->nvml_enabled = false;
    pb->sampling_thread = pthread_t();
    pb->stop_sampling = true;
    pb->sampling_active = false;
    pb->sampling_thread_started = false;
    pb->sampling_failed = false;
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

static void power_buffer_samples_clear(PowerBuffer* pb) {
    if (pb == nullptr) return;
    pb->samples.clear();
    pb->sampling_failed = false;
}

static void power_buffer_start_sampling(PowerBuffer* pb) {
    if (pb == nullptr || !pb->nvml_enabled) return;
    if (pb->sampling_active) return;
    pb->stop_sampling = false;
    pb->sampling_active = true;
    power_buffer_sample_once(pb);
    if (pthread_create(&pb->sampling_thread, nullptr, power_buffer_sampling_main, pb) == 0) {
        pb->sampling_thread_started = true;
    }
}

static void power_buffer_stop_sampling(PowerBuffer* pb) {
    if (pb == nullptr || !pb->nvml_enabled || !pb->sampling_active) return;
    pb->stop_sampling = true;
    if (pb->sampling_thread_started) {
        pthread_join(pb->sampling_thread, nullptr);
        pb->sampling_thread_started = false;
    }
    power_buffer_sample_once(pb);
    pb->sampling_active = false;
}

static bool power_buffer_capture_valid(const PowerBuffer* pb) {
    return pb != nullptr && pb->nvml_enabled && !pb->sampling_failed && pb->samples.size() >= 2;
}

static double power_buffer_energy_joules(const PowerBuffer* pb) {
    if (!power_buffer_capture_valid(pb)) return 0.0;
    double energy_j = 0.0;
    for (size_t i = 1; i < pb->samples.size(); ++i) {
        const PowerSample& a = pb->samples[i - 1];
        const PowerSample& b = pb->samples[i];
        if (b.timestamp_ns <= a.timestamp_ns) continue;
        const double dt_s = static_cast<double>(b.timestamp_ns - a.timestamp_ns) / 1e9;
        const double p0_w = static_cast<double>(a.power_mw) / 1000.0;
        const double p1_w = static_cast<double>(b.power_mw) / 1000.0;
        energy_j += 0.5 * (p0_w + p1_w) * dt_s;
    }
    return std::isfinite(energy_j) ? energy_j : 0.0;
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
