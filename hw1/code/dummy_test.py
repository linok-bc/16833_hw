import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Your samplers (copy-paste yours here)
# -----------------------------
class Resampling:
    def multinomial_sampler(self, X_bar):
        w = X_bar[:, 3]
        w = w / np.sum(w)

        N = X_bar.shape[0]
        indices = np.random.choice(N, size=N, p=w)

        X_bar_resampled = X_bar[indices].copy()
        X_bar_resampled[:, 3] = 1 / N
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        N = X_bar.shape[0]
        w = X_bar[:, 3]
        w = w / np.sum(w)
        cdf = np.cumsum(w)

        r = np.random.uniform(0, 1.0 / N)

        X_bar_resampled = np.zeros_like(X_bar)
        curr = 0
        for k in range(N):
            u = r + k / N
            while curr < N - 1 and u > cdf[curr]:
                curr += 1
            X_bar_resampled[k] = X_bar[curr]

        X_bar_resampled[:, 3] = 1 / N
        return X_bar_resampled


# -----------------------------
# Create dummy particles
# -----------------------------
def create_dummy_particles(N=200):
    """
    Create particles with:
    - two clusters
    - uneven weights
    """
    particles = np.zeros((N, 4))

    # Cluster 1 (high weight)
    n1 = int(0.7 * N)
    particles[:n1, 0] = np.random.normal(0, 0.5, n1)   # x
    particles[:n1, 1] = np.random.normal(0, 0.5, n1)   # y

    # Cluster 2 (low weight)
    n2 = N - n1
    particles[n1:, 0] = np.random.normal(5, 0.5, n2)
    particles[n1:, 1] = np.random.normal(5, 0.5, n2)

    # headings
    particles[:, 2] = np.random.uniform(-np.pi, np.pi, N)

    # weights (biased toward cluster 1)
    weights = np.ones(N)
    weights[:n1] *= 5   # heavier cluster
    weights[n1:] *= 1

    weights = weights / np.sum(weights)
    particles[:, 3] = weights

    return particles


# -----------------------------
# Diagnostics
# -----------------------------
def print_stats(name, particles):
    w = particles[:, 3]
    print(f"\n{name} stats:")
    print(f"  Weight sum: {np.sum(w):.6f}")
    print(f"  Unique particles: {len(np.unique(particles[:, :2], axis=0))}")


# -----------------------------
# Visualization
# -----------------------------
def plot_particles(original, multi, lowvar):
    plt.figure(figsize=(15, 4))

    titles = ["Original", "Multinomial", "Low Variance"]
    datasets = [original, multi, lowvar]

    for i, (title, data) in enumerate(zip(titles, datasets)):
        plt.subplot(1, 3, i + 1)
        plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.6)
        plt.title(title)
        plt.axis("equal")

    plt.tight_layout()
    plt.show()


# -----------------------------
# Main test
# -----------------------------
if __name__ == "__main__":
    np.random.seed(0)

    resampler = Resampling()

    # Create dummy particles
    X = create_dummy_particles(N=300)

    print("Original distribution:")
    print(f"  Weight sum: {np.sum(X[:,3]):.6f}")

    # Run samplers
    X_multi = resampler.multinomial_sampler(X)
    X_lowvar = resampler.low_variance_sampler(X)

    # Diagnostics
    print_stats("Multinomial", X_multi)
    print_stats("Low variance", X_lowvar)

    # Plot
    plot_particles(X, X_multi, X_lowvar)
