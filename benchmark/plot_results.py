import matplotlib.pyplot as plt
import csv


def load_results(filename):
    loads = []
    throughput = []
    latency = []

    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            loads.append(row["Load"])
            throughput.append(float(row["Throughput"]))
            latency.append(float(row["AvgLatency"]))

    return loads, throughput, latency


loads, micro_thr, micro_lat = load_results("results_micro.csv")
_, mono_thr, mono_lat = load_results("results_mono.csv")

plt.figure()
plt.plot(loads, micro_lat, label="Microservices")
plt.plot(loads, mono_lat, label="Monolith")
plt.title("Average Latency Comparison")
plt.ylabel("Latency (seconds)")
plt.legend()
plt.savefig("latency.png")

plt.figure()
plt.plot(loads, micro_thr, label="Microservices")
plt.plot(loads, mono_thr, label="Monolith")
plt.title("Throughput Comparison")
plt.ylabel("Requests per Second")
plt.legend()
plt.savefig("throughput.png")

print("Plots generated: latency.png, throughput.png")