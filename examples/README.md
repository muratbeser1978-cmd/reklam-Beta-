# Examples / Örnekler

Bu klasörde simülasyonun nasıl çalıştırılacağına dair örnekler bulunmaktadır.

## Örnekler

### 1. `basic_simulation.py` - Temel Simülasyon
En basit örnek. Tek bir simülasyon çalıştırır ve sonucu görselleştirir.

```bash
python examples/basic_simulation.py
```

**Çıktı:**
- `outputs/examples/basic_trajectory.pdf` - Pazar payı grafiği

**Ne Yapıyor:**
- 1000 tüketici, 500 zaman adımı
- γ = 0.75 (γ* = 0.706'dan büyük → tek denge)
- Trajectory plot oluşturur

---

### 2. `bifurcation_analysis.py` - Bifurcation Analizi
Kritik nokta γ* = 12/17 civarında bifurcation analizi yapar.

```bash
python examples/bifurcation_analysis.py
```

**Çıktı:**
- `outputs/examples/bifurcation_diagram.pdf` - Bifurcation diyagramı
- `outputs/examples/bifurcation_amplitude.pdf` - Amplitude scaling

**Ne Yapıyor:**
- Farklı γ değerlerinde denge noktalarını bulur
- γ < γ*: 3 denge (2 asimetrik + 1 stabil olmayan simetrik)
- γ > γ*: 1 denge (stabil simetrik)
- Supercritical pitchfork bifurcation görselleştirir

---

### 3. `parameter_sweep.py` - Parametre Taraması
Paralel olarak birden fazla γ değerinde simülasyon çalıştırır.

```bash
python examples/parameter_sweep.py
```

**Çıktı:**
- `outputs/examples/sweep/*.npz` - Her simülasyon için veri
- `outputs/examples/sweep/*.json` - Provenance metadata

**Ne Yapıyor:**
- 10 farklı γ değeri × 3 farklı seed = 30 simülasyon
- 4 CPU core kullanarak paralel çalışır
- Her sonucu provenance bilgisiyle kaydeder (git commit, environment, vb.)

---

### 4. `full_analysis.py` - Tam Analiz
Komple analiz pipeline'ı: simülasyon → denge → istatistik → refah → grafikler.

```bash
python examples/full_analysis.py
```

**Çıktı:**
- `outputs/examples/full_analysis/trajectory.pdf`
- `outputs/examples/full_analysis/potential_slices.pdf`
- `outputs/examples/full_analysis/welfare_trajectory.pdf`

**Ne Yapıyor:**
1. Simülasyon çalıştırır (γ = 0.65 < γ*)
2. Denge noktalarını bulur
3. İstatistiksel analiz (mean, variance, autocorrelation, skewness)
4. Refah analizi (optimal vs equilibrium)
5. 3 farklı görselleştirme oluşturur

---

### 5. `visualization_gallery.py` - Görselleştirme Galerisi
Tüm 13 görselleştirme fonksiyonunu gösterir.

```bash
python examples/visualization_gallery.py
```

**Çıktı:**
- 10 farklı PDF grafiği `outputs/examples/gallery/` içinde

**Ne Yapıyor:**
- Tek trajectory
- Çoklu trajectories
- Variance bands
- Bifurcation diagram
- Bifurcation amplitude
- 3D potential landscape
- 2D potential contour
- Potential slices
- Potential gradient
- Welfare loss

⚠️ **Not:** Bu örnek en uzun çalışandır (~2-3 dakika)

---

## Hızlı Başlangıç

### İlk kez çalıştırıyorsanız:

```bash
# 1. Virtual environment oluştur
python3 -m venv venv
source venv/bin/activate

# 2. Bağımlılıkları yükle
pip install -r requirements.txt

# 3. Basit örneği çalıştır
python examples/basic_simulation.py

# 4. Sonucu kontrol et
open outputs/examples/basic_trajectory.pdf
```

### Tüm örnekleri çalıştır:

```bash
# Her birini sırayla
python examples/basic_simulation.py
python examples/bifurcation_analysis.py
python examples/parameter_sweep.py
python examples/full_analysis.py
python examples/visualization_gallery.py  # En uzun süren
```

---

## Çıktı Dizini Yapısı

```
outputs/
├── examples/
│   ├── basic_trajectory.pdf
│   ├── bifurcation_diagram.pdf
│   ├── bifurcation_amplitude.pdf
│   ├── sweep/
│   │   ├── sweep_gamma0.6000_seed00042.npz
│   │   ├── sweep_gamma0.6000_seed00042.json
│   │   └── ... (30 files)
│   ├── full_analysis/
│   │   ├── trajectory.pdf
│   │   ├── potential_slices.pdf
│   │   └── welfare_trajectory.pdf
│   └── gallery/
│       ├── 01_trajectory.pdf
│       ├── 02_multiple_trajectories.pdf
│       └── ... (10 files)
```

---

## Kendi Analizini Yap

Örnekleri temel alarak kendi analizini yapabilirsin:

```python
from src.simulation.config import SimulationConfig
from src.simulation.engine import SimulationEngine
from src.visualization import plot_trajectory

# Konfigürasyonu değiştir
config = SimulationConfig(
    N=2000,           # Daha fazla tüketici
    T=1000,           # Daha uzun simülasyon
    gamma=0.65,       # Farklı γ değeri
    beta=0.95,        # Farklı churn rate
    seed=123,         # Farklı seed
    p_init=0.7        # Farklı başlangıç noktası
)

# Çalıştır
engine = SimulationEngine()
result = engine.run(config)

# Görselleştir
plot_trajectory(result, save_path="my_custom_plot.pdf")
```

---

## Yardım

Sorun yaşarsan veya daha fazla örnek istersen:
1. Ana README.md'yi kontrol et
2. Kod içindeki docstring'leri oku
3. Issue aç: https://github.com/your-repo/issues
