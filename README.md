Bu proje Google Colab üzerinden çalıştırılmak üzere optimize edilmiştir. 
Lütfen `Google Drive` ile model dosyalarınızı bağlayın ve `hibrit_model.py` içerisindeki yolları güncelleyin.


Tabii! İşte projeniz için bir **README** dosyası örneği:

---

# **Adli Psikolojik Risk Tespit Ajanı**

## **Proje Tanımı**
Adli Psikolojik Risk Tespit Ajanı, doğal dil işleme (NLP) ve makine öğrenimi teknikleri kullanarak yazılı ifadelerdeki psikolojik risk faktörlerini analiz etmek için geliştirilmiş bir modeldir. Bu model, metinlerden duygu durumu, psikolojik hastalık belirtileri, manipülasyon izleri, empati seviyesi ve tehdit seviyesini tespit ederek, potansiyel psikolojik riskleri sınıflandırır.

## **Kullanılan Teknolojiler ve Modeller**
- **RoBERTa Modeli**: Bu model, duygusal ve psikolojik riskleri tespit etmek için eğitilmiş bir dil modeli kullanır.
- **BERT Modeli**: Metin sınıflandırması ve psikolojik risk tespiti için bir başka derin öğrenme modeli.
- **LLaMA API**: Meta LLaMA modelini kullanarak derinlemesine psikolojik analiz yapılır ve metinlerin psikolojik durumları hakkında daha kapsamlı bilgiler sağlanır.

## **Model Özellikleri**
- **Genel Duygu Tonu**: Metnin pozitif, negatif veya nötr olup olmadığını analiz eder.
- **Risk Tipi**: Metindeki duygusal ve psikolojik durumları değerlendirir (Öfke, Depresyon, Tehdit, Manipülasyon, Kişilik Bozukluğu vb.).
- **Tehdit Seviyesi**: 0-10 arasında bir tehdit seviyesi belirler.
- **Manipülasyon Belirtileri**: Metinde manipülasyon izlerinin olup olmadığını belirler.
- **Empati Seviyesi**: Düşük, orta veya yüksek empati seviyeleri belirler.
- **Profesyonel Açıklama**: Psikolojik analiz ve risk değerlendirmeleri hakkında profesyonel açıklamalar sunar.

## **Modelin Kullanımı**

### **Modelin Eğitilmesi**
Model, aşağıdaki adımlar ile eğitilebilir:
1. **Veri Hazırlığı**: Etiketli psikolojik risk içeren metinler (örneğin, anksiyete, depresyon, intihar eğilimi vb.) kullanılmalıdır.
2. **Model Eğitimi**: Hugging Face üzerinde mevcut BERT ve RoBERTa modelleri fine-tune edilerek kullanılabilir.
3. **Test ve Doğrulama**: Modelin doğruluğunu test etmek için uygun test verileri kullanılmalıdır.

### **Kodun Çalıştırılması**
Aşağıdaki adımlar ile modeli çalıştırabilirsiniz:

1. **Gerekli Kütüphaneleri Yükleyin**:
   ```bash
   pip install transformers torch numpy requests
   ```

2. **Modeli Yükleyin ve Test Edin**:
   ```python
   from transformers import AutoTokenizer, AutoModelForSequenceClassification
   import torch
   import numpy as np

   # Modeli yükleme fonksiyonları
   def load_roberta_model(path="path_to_roberta_model"):
       tokenizer = AutoTokenizer.from_pretrained(path)
       model = AutoModelForSequenceClassification.from_pretrained(path)
       return tokenizer, model

   def load_bert_model(path="path_to_bert_model"):
       tokenizer = AutoTokenizer.from_pretrained(path)
       model = AutoModelForSequenceClassification.from_pretrained(path)
       return tokenizer, model

   # Metin üzerinden tahmin yapma
   def predict_model(text, tokenizer, model):
       model.eval()
       inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
       with torch.no_grad():
           outputs = model(**inputs)
           logits = outputs.logits
           probs = torch.nn.functional.softmax(logits, dim=1)
       probs = probs.cpu().numpy().flatten()
       return {label: float(prob) for label, prob in zip(label_list, probs)}

   # Model çıktısını almak için örnek kullanım
   text = "I feel like I don’t matter to anyone. Life is meaningless."
   result = hybrid_analysis(text, roberta_tok, roberta_model, bert_tok, bert_model)
   print("🧠 Final Risk Label:", result["final_label"])
   print("📊 RoBERTa Probs:", result["roberta_probs"])
   print("📊 BERT Probs:", result["bert_probs"])
   print("🧾 LLaMA Analysis:", result["llama_analysis"])
   ```

3. **LLaMA API İle Çalıştırma**:
   - **API Anahtarınız** gereklidir.
   - LLaMA modeline istek göndermek için `requests` kütüphanesini kullanarak modelden analiz elde edebilirsiniz.

## **Proje Sonuçları**
Proje, kullanıcıların metinlerdeki psikolojik riskleri belirlemelerine yardımcı olur. Bu analiz, özellikle adli psikoloji, psikoterapi ve kişisel gelişim alanlarında kullanılabilir. Ayrıca, bu modelin çıktıları, kriz durumları için erken uyarı mekanizmaları geliştirmek için değerlidir.

## **Sonuçlar ve Değerlendirme**
Model, üç ana kaynaktan (RoBERTa, BERT ve LLaMA) gelen çıktıları harmanlayarak daha doğru ve kapsamlı risk değerlendirmeleri sağlar. Bu model, psikolojik destek isteyen bireylerin hızlı bir şekilde tespit edilmesine yardımcı olabilir.

## **Gelecek Adımlar**
- **Model Geliştirme**: Daha fazla etiketli veriye dayalı olarak modelin doğruluğunu artırma.
- **Çok Dilli Destek**: Farklı dillerde metinleri analiz edebilme yeteneği eklenebilir.
- **Gerçek Zamanlı İzleme**: Canlı metin akışlarına dayanarak psikolojik risk tespiti yapabilen sistemler geliştirme.


