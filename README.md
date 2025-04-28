 MindWatch: Adli Psikolojik Risk Analiz Ajanı
 Proje Tanımı
MindWatch, doğal dil işleme (NLP) ve makine öğrenimi teknikleri kullanarak yazılı metinlerdeki psikolojik risk faktörlerini analiz eden bir sistemdir.
Bu model, depresyon, intihar eğilimi, tehdit, öfke, manipülasyon gibi psikolojik durumları tespit eder ve her bir metne profesyonel bir risk değerlendirmesi sunar.
Sistem, BERT, RoBERTa ve Meta LLaMA 70B API çıktılarının hibrit birleştirilmesiyle daha doğru analizler üretir.

 Kullanılan Teknolojiler
BERT Modeli: Psikolojik risk sınıflandırması için fine-tune edilmiş.

RoBERTa Modeli: Duygusal analiz ve risk tespiti için optimize edilmiş.

LLaMA 70B API: Derinlemesine psikolojik değerlendirme ve açıklamalı risk analizi.

Gradio Arayüzü: Hızlı kullanıcı arayüzü ile Colab uyumlu çalışma.

Google Drive Entegrasyonu: Model dosyaları Drive üzerinden yüklenir.

 Modelin Sağladığı Analizler
Genel Duygu Tonu: Pozitif / Negatif / Nötr sınıflaması.

Risk Tipi Belirleme: (Depresyon, İntihar, Öfke, Manipülasyon, Tehdit vb.)

Tehdit Seviyesi: 0-10 arası puanlama.

Manipülasyon Belirtileri: Var / Yok tespiti.

Empati Seviyesi: Düşük / Orta / Yüksek.

Profesyonel Açıklamalar: LLaMA API üzerinden detaylı psikolojik raporlar.

 Kurulum ve Kullanım
1. Gerekli Kütüphaneleri Yükleyin
Colab veya lokal ortamda:

bash
Kopyala
Düzenle
pip install transformers torch numpy requests gradio
2. Google Drive'ı Bağlayın
Model ağırlıkları Drive'da saklandığı için:

python
Kopyala
Düzenle
from google.colab import drive
drive.mount('/content/drive')
Ardından hibrit_model.py dosyasındaki model yükleme yollarını kendi Drive yolunuza göre düzenleyin.

3. Uygulamayı Başlatın
Gradio arayüzünü çalıştırmak için:

python
Kopyala
Düzenle
import gradio as gr
from hibrit_model import hybrid_analysis

def analyze_text(text):
    result = hybrid_analysis(text)
    return f"BERT: {result['bert_result']['final_decision']} ({result['bert_result']['confidence']*100:.1f}%)\n" \
           f"RoBERTa: {result['roberta_result']['final_decision']} ({result['roberta_result']['confidence']*100:.1f}%)\n" \
           f"Ensemble: {result['ensemble_result']['final_decision']} ({result['ensemble_result']['confidence']*100:.1f}%)\n" \
           f"LLaMA Label: {result['llama_label']}"

interface = gr.Interface(fn=analyze_text, inputs="text", outputs="text", title="MindWatch: Psychological Risk Analyzer")
interface.launch(share=True)
share=True dersen link verir, herkes tarayıcıdan ulaşabilir.

 LLaMA API Kullanımı
LLaMA analizleri için IO.net veya Intelligence API'den alınmış bir API anahtarınız olmalıdır.

API anahtarınızı hibrit_model.py içindeki ask_llama(text) fonksiyonunda Authorization kısmına ekleyin.

Her analiz sırasında 7 farklı prompt LLaMA'ya gönderilerek profesyonel bir psikolojik analiz oluşturulur.

 Örnek Analiz Akışı
Verilen bir kullanıcı metni için:


Model	Sonuç	Güven Skoru (%)
BERT	Depresyon	85.4%
RoBERTa	Anksiyete	78.2%
Ensemble	Depresyon	91.2%
LLaMA	Riskli (Depresyon + İntihar Eğilimi)	
Kritik risk algılandığında sistem ayrıca ⚠️ alarm verir.

 Proje Sonuçları
Model, psikolojik riskleri yüksek doğrulukla tespit etmektedir.

Hibrit yapısı sayesinde tek bir modelin hatalarına düşmeden sağlam bir değerlendirme sağlar.

Derinlemesine LLaMA analizleri ile sadece sınıflama değil, neden riskli olduğunu açıklayan profesyonel çıktılar üretir.

 Gelecek Geliştirmeler
Çok Dilli Destek: İngilizce dışındaki diller için model genişletme.

Gerçek Zamanlı İzleme: Online sohbet platformları veya sosyal medya gibi yerlerde psikolojik risklerin izlenmesi.

Otomatik Kriz Müdahale Önerileri: Kritik risk algılandığında otomatik destek veya müdahale planları sunulması.

ses ve görüntü verileriyle modeli besleme


 Kapanış
MindWatch, psikolojik risk analizinde hem akademik, hem adli kullanım için yüksek potansiyelli bir destek aracıdır.
Yarışmalarda, klinik destek uygulamalarında ve adli vaka analizlerinde rahatlıkla kullanılabilir.
