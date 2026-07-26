// Sonda de energia in-process para Fase 4 (Camino A): dos lecturas de un
// contador de energia MONOTONICO en los mismos instantes begin/end que ya
// delimitan emit_csv_region_marker en cada benchmark. Dos lecturas y una
// resta bastan y son exactas; no hay sampler ni hilo de integracion.
//
// Opt-in por macro: sin -DUSE_NVML_TELEMETRY, EnergyProbe es un stub que no
// mide (result() siempre invalido -> "NA" en el CSV), este header no incluye
// <nvml.h>/<dirent.h> y el build no requiere -lnvidia-ml. Con la macro,
// mide GPU via NVML (nvmlDeviceGetTotalEnergyConsumption) y CPU via RAPL
// (/sys/class/powercap), degradando a "solo GPU" si RAPL no es legible.
//
// IMPORTANT -- a diferencia de common.cuh (que se incluye DENTRO del
// `namespace { ... }` anonimo de cada .cu para conservar enlace interno),
// este header debe incluirse a nivel GLOBAL, antes de cualquier
// `namespace { ... }`: <nvml.h> y <dirent.h> declaran simbolos `extern "C"`
// de las bibliotecas del sistema (nvmlInit_v2, opendir, etc.); si el
// #include cae dentro de un namespace anonimo, esas declaraciones quedan con
// enlace interno (regla de C++, namespace.unnamed) y el linker deja de
// resolverlas contra libnvidia-ml.so / libc.
#pragma once

#include <cstdio>
#include <string>

// Muestra de energia/potencia de una region begin/end. energy_j/avg_power_w/
// edp son de GPU (las 3 columnas que existen en el CSV de Fase 3);
// cpu_energy_j se mide pero no tiene columna propia todavia.
struct EnergySample {
    bool gpu_valid = false;
    bool cpu_valid = false;
    double wall_s = 0.0;
    double energy_j = 0.0;      // GPU, NVML: mJ acumulados (end - begin) / 1000.
    double avg_power_w = 0.0;   // GPU: energy_j / wall_s.
    double cpu_energy_j = 0.0;  // CPU, RAPL: uJ acumulados (end - begin) / 1e6.
    double edp = 0.0;           // GPU: energy_j * wall_s (energy-delay product).
};

// Formatea v como escalar (6 cifras cientificas, igual convencion que
// fmt_sci() en los .cu) si valid, o "NA" si no. Evita repetir el ternario
// `valid ? fmt_sci(v) : "NA"` en cada sitio que arma una fila de CSV.
inline std::string energy_field(bool valid, double v) {
    if (!valid) return "NA";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6e", v);
    return buf;
}

#ifdef USE_NVML_TELEMETRY

#include <cctype>
#include <chrono>
#include <cstring>
#include <vector>

#include <dirent.h>
#include <nvml.h>

#include <cuda_runtime.h>

// Lee energia acumulada de RAPL (paquete/dram del host) sumando los
// dominios de NIVEL SUPERIOR bajo /sys/class/powercap, es decir
// "intel-rapl:N" (paquete N). Los subdominios "intel-rapl:N:M"
// (core/uncore/dram) se excluyen a proposito: estan incluidos dentro del
// total del paquete que los contiene, sumarlos aparte contaria esa energia
// dos veces.
class RaplReader {
public:
    RaplReader() { discover_domains(); }

    bool available() const { return !domains_.empty(); }

    // Snapshot en uJ, un valor por dominio de nivel superior. false si algun
    // dominio no pudo leerse en este instante (p.ej. permiso revocado a
    // mitad de la corrida); el llamador debe tratar la ventana completa como
    // no medida en ese caso.
    bool snapshot(std::vector<unsigned long long>& out) const {
        out.resize(domains_.size());
        for (size_t i = 0; i < domains_.size(); ++i) {
            if (!read_u64(domains_[i].energy_path, out[i])) return false;
        }
        return true;
    }

    // Delta en J entre dos snapshots del mismo tamano que domains_. Corrige
    // wraparound POR DOMINIO (cada uno envuelve en su propio
    // max_energy_range_uj, tipicamente distinto entre paquete y dram):
    // sumar los totales begin/end y restar al final asumiria un unico rango
    // global y subestimaria la energia si un dominio envolvio y otro no.
    double delta_j(const std::vector<unsigned long long>& begin,
                   const std::vector<unsigned long long>& end) const {
        double total_uj = 0.0;
        for (size_t i = 0; i < domains_.size(); ++i) {
            const unsigned long long b = begin[i];
            const unsigned long long e = end[i];
            const unsigned long long range = domains_[i].max_range_uj;
            const unsigned long long d = (e >= b) ? (e - b) : (range - b + e);
            total_uj += static_cast<double>(d);
        }
        return total_uj / 1e6;
    }

private:
    struct Domain {
        std::string energy_path;
        unsigned long long max_range_uj = 0;
    };
    std::vector<Domain> domains_;

    static bool read_u64(const std::string& path, unsigned long long& out) {
        std::FILE* f = std::fopen(path.c_str(), "r");
        if (!f) return false;
        const int n = std::fscanf(f, "%llu", &out);
        std::fclose(f);
        return n == 1;
    }

    // Enumera /sys/class/powercap con <dirent.h> (no hay ningun glob
    // portable mas simple para esto) y se queda solo con entradas
    // "intel-rapl:<digitos>" -- descarta subdominios ("intel-rapl:0:1") y
    // cualquier otro tipo de dominio powercap presente en el sistema.
    // Si energy_uj no es legible (mitigacion PLATYPUS en kernels recientes:
    // solo root, o un usuario con CAP_PERFMON/regla udev explicita puede
    // leerlo) el dominio se omite en vez de abortar; si terminan
    // omitiendose todos, available() devuelve false y EnergyProbe degrada a
    // "solo GPU" sin tocar el resto de la medicion.
    void discover_domains() {
        DIR* dir = opendir("/sys/class/powercap");
        if (!dir) return;
        struct dirent* ent;
        const std::string prefix = "intel-rapl:";
        while ((ent = readdir(dir)) != nullptr) {
            const std::string name = ent->d_name;
            if (name.rfind(prefix, 0) != 0) continue;
            const std::string suffix = name.substr(prefix.size());
            if (suffix.empty() || suffix.find(':') != std::string::npos) continue;
            bool all_digits = true;
            for (char c : suffix) {
                if (!std::isdigit(static_cast<unsigned char>(c))) { all_digits = false; break; }
            }
            if (!all_digits) continue;

            Domain d;
            d.energy_path = "/sys/class/powercap/" + name + "/energy_uj";
            const std::string range_path = "/sys/class/powercap/" + name + "/max_energy_range_uj";
            unsigned long long probe = 0;
            if (!read_u64(d.energy_path, probe)) continue;
            if (!read_u64(range_path, d.max_range_uj)) continue;
            domains_.push_back(std::move(d));
        }
        closedir(dir);
    }
};

// Handle NVML perezoso: nvmlInit_v2 + nvmlDeviceGetHandleByIndex solo la
// primera vez que hace falta. La inicializacion de una variable local
// `static` esta garantizada thread-safe desde C++11 ("magic statics"), asi
// que esto es seguro aunque EnergyProbe se llegue a usar desde mas de un
// hilo. No hay nvmlShutdown explicito: el proceso vive lo que dura una
// corrida de benchmark y el driver de NVIDIA libera el contexto NVML al
// salir del proceso.
inline bool nvml_device_handle(nvmlDevice_t& out_handle) {
    static bool initialized = false;
    static bool ok = false;
    static nvmlDevice_t handle{};
    if (!initialized) {
        initialized = true;
        // Asume que el indice de enumeracion de NVML coincide con el
        // dispositivo CUDA activo -- cierto salvo con CUDA_VISIBLE_DEVICES
        // reordenando el mapeo o con MIG particionando la GPU. PACCA asigna
        // un A100 completo por job (sin MIG), asi que el indice 0 de NVML es
        // ese mismo dispositivo.
        int cuda_dev = 0;
        cudaGetDevice(&cuda_dev);
        ok = (nvmlInit_v2() == NVML_SUCCESS) &&
             (nvmlDeviceGetHandleByIndex(static_cast<unsigned>(cuda_dev), &handle) == NVML_SUCCESS);
    }
    if (ok) out_handle = handle;
    return ok;
}

class EnergyProbe {
public:
    // Se llama justo DESPUES de emit_csv_region_marker(route, "begin"),
    // dentro de la misma ventana que ese marcador delimita (ver comentario
    // en cada benchmark_gpu_*). Incluye entonces todo lo que hay dentro de
    // esa ventana, incluidas las copias D2H de checkpoint: son parte real
    // del costo energetico de la corrida, y separarlas exigiria pausar
    // tambien la sonda de energia alrededor de cada checkpoint (mismo
    // problema que ya resuelve CudaEventTimer con start/stop, pero
    // duplicado); no vale la pena la complejidad extra por una fraccion de
    // ventana que ya se reporta aparte en t_ms_iter_ckpt.
    void begin() {
        wall_t0_ = std::chrono::steady_clock::now();

        nvmlDevice_t dev;
        gpu_ok_begin_ = nvml_device_handle(dev) &&
            (nvmlDeviceGetTotalEnergyConsumption(dev, &gpu_mj_begin_) == NVML_SUCCESS);

        cpu_ok_begin_ = rapl_.available() && rapl_.snapshot(cpu_uj_begin_);
    }

    void end() {
        wall_t1_ = std::chrono::steady_clock::now();

        nvmlDevice_t dev;
        gpu_ok_end_ = gpu_ok_begin_ && nvml_device_handle(dev) &&
            (nvmlDeviceGetTotalEnergyConsumption(dev, &gpu_mj_end_) == NVML_SUCCESS);

        cpu_ok_end_ = cpu_ok_begin_ && rapl_.snapshot(cpu_uj_end_);
    }

    // avg_power_w y edp se calculan sobre wall_s (pared, begin a end), NO
    // sobre t_ms_total (suma de kernels GPU): la ventana de energia mide
    // consumo real de principio a fin de la region, que incluye tiempo de
    // GPU ociosa (D2H de checkpoint, overhead de lanzamiento) tan real desde
    // el medidor de potencia como el tiempo de computo. Dividir por
    // t_ms_total en cambio inflaria avg_power_w artificialmente (mismos J
    // repartidos entre menos segundos) y no es lo que EDP mide en la
    // literatura (energy-delay product usa el delay de punta a punta de la
    // tarea, no solo su porcion de computo).
    EnergySample result() const {
        EnergySample s;
        s.wall_s = std::chrono::duration<double>(wall_t1_ - wall_t0_).count();

        s.gpu_valid = gpu_ok_begin_ && gpu_ok_end_ && gpu_mj_end_ >= gpu_mj_begin_;
        if (s.gpu_valid) {
            s.energy_j = static_cast<double>(gpu_mj_end_ - gpu_mj_begin_) / 1000.0;
            s.avg_power_w = (s.wall_s > 0.0) ? s.energy_j / s.wall_s : 0.0;
            s.edp = s.energy_j * s.wall_s;
        }

        s.cpu_valid = cpu_ok_begin_ && cpu_ok_end_;
        if (s.cpu_valid) {
            s.cpu_energy_j = rapl_.delta_j(cpu_uj_begin_, cpu_uj_end_);
        }
        return s;
    }

private:
    std::chrono::steady_clock::time_point wall_t0_{};
    std::chrono::steady_clock::time_point wall_t1_{};

    bool gpu_ok_begin_ = false;
    bool gpu_ok_end_ = false;
    unsigned long long gpu_mj_begin_ = 0;
    unsigned long long gpu_mj_end_ = 0;

    RaplReader rapl_;
    bool cpu_ok_begin_ = false;
    bool cpu_ok_end_ = false;
    std::vector<unsigned long long> cpu_uj_begin_;
    std::vector<unsigned long long> cpu_uj_end_;
};

#else  // !USE_NVML_TELEMETRY

// Stub: no mide nada, result() siempre invalido -> energy_field() imprime
// "NA" para las 3 columnas. No incluye <nvml.h>/<dirent.h> ni requiere
// -lnvidia-ml, para no romper el build por defecto.
class EnergyProbe {
public:
    void begin() {}
    void end() {}
    EnergySample result() const { return EnergySample{}; }
};

#endif  // USE_NVML_TELEMETRY
