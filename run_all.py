#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ricci-NN Projesini Baştan Sona Çalıştıran Master Script
"""
import os
import sys
import subprocess

def run_script(script_name, description):
    """Bir Python script'ini çalıştırır ve sonucu gösterir"""
    print("\n" + "="*70)
    print(f"📍 {description}")
    print("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ {script_name} başarıyla tamamlandı!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ HATA: {script_name} çalıştırılırken hata oluştu!")
        print(f"Hata detayı: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ HATA: {script_name} bulunamadı!")
        return False

def check_file_exists(filename):
    """Dosyanın varlığını kontrol eder"""
    return os.path.exists(filename)

def main():
    print("🚀 Ricci-NN Projesi Otomatik Çalıştırma")
    print("="*70)
    
    # Çalışma dizinini ayarla
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📂 Çalışma dizini: {script_dir}\n")
    
    # Adım 1: Veri setini indir
    if not check_file_exists("fashion-mnist_train.csv") or not check_file_exists("fashion-mnist_test.csv"):
        if not run_script("download_dataset.py", "Adım 1/4: Fashion MNIST veri seti indiriliyor"):
            print("\n⚠️  Veri seti indirme başarısız. Devam etmek için bu dosyalar gerekli.")
            return
    else:
        print("\n✓ Fashion MNIST veri seti zaten mevcut, atlıyorum...")
    
    # Adım 2: Label 5 ve 9'u çıkar
    if not run_script("fmnist_extraction.py", "Adım 2/4: Label 5 ve 9 çıkarılıyor"):
        print("\n⚠️  Veri çıkarma başarısız.")
        return
    
    # Adım 3: Model eğitimi
    if not check_file_exists("model_predict.npy"):
        print("\n⏰ DİKKAT: Model eğitimi 10-20 dakika sürebilir...")
        response = input("Devam etmek istiyor musunuz? (E/H): ")
        if response.lower() not in ['e', 'evet', 'y', 'yes']:
            print("❌ İşlem kullanıcı tarafından iptal edildi.")
            return
        
        if not run_script("training.py", "Adım 3/4: DNN modelleri eğitiliyor"):
            print("\n⚠️  Model eğitimi başarısız.")
            return
    else:
        print("\n✓ Eğitilmiş modeller mevcut, atlıyorum...")
    
    # Adım 4: Ricci eğriliği analizi
    if not run_script("knn.py", "Adım 4/4: Ricci eğriliği hesaplanıyor ve analiz yapılıyor"):
        print("\n⚠️  Ricci analizi başarısız.")
        return
    
    print("\n" + "="*70)
    print("🎉 TÜM ADIMLAR BAŞARIYLA TAMAMLANDI!")
    print("="*70)
    print("\n📊 Oluşturulan dosyalar:")
    
    output_files = [
        "fashion-mnist_train.csv",
        "fashion-mnist_test.csv",
        "activation_model0.h5",
        "activation_model1.h5",
        "activation_model2.h5",
        "model_predict.npy",
        "accuracy.npy",
        "x_test.csv",
        "y_test.csv"
    ]
    
    for f in output_files:
        if check_file_exists(f):
            size = os.path.getsize(f) / (1024*1024)  # MB
            print(f"  ✓ {f:<30} ({size:.2f} MB)")
    
    print("\n📁 data_fmnist/ klasöründeki dosyalar:")
    if os.path.exists("data_fmnist"):
        for f in os.listdir("data_fmnist"):
            if f.endswith('.csv'):
                full_path = os.path.join("data_fmnist", f)
                size = os.path.getsize(full_path) / (1024*1024)
                print(f"  ✓ {f:<30} ({size:.2f} MB)")

if __name__ == "__main__":
    main()

