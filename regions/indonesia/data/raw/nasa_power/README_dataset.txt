DATA CUACA NASA POWER 1981-2025
20 lokasi: Batam, Semarang, Wonosobo, Kupang, Ambon, Banda_Aceh, Medan, Padang, Pekanbaru, Palembang, Jakarta, Bandung, Surabaya, Denpasar, Mataram, Pontianak, Banjarmasin, Makassar, Manado, Jayapura. Setiap lokasi dipisahkan dalam folder sendiri.
Cakupan variabel: seluruh 12 variabel yang dibahas dalam percakapan, bukan seluruh katalog NASA POWER.
Periode kalender: 1981-01-01 sampai 2025-12-31, 16.436 hari dan 540 bulan per lokasi.
Sumber: https://power.larc.nasa.gov/docs/services/api/temporal/daily/
Estimasi berbasis grid pada koordinat representatif kota; bukan pengukuran langsung BMKG atau rata-rata wilayah administratif.
LST (Local Solar Time) NASA POWER, bukan zona waktu administratif WIB/WIT.

KAMUS VARIABEL HARIAN
PRECTOTCORR: curah hujan terkoreksi (mm/hari).
T2M: suhu udara rata-rata 2 m (C).
T2M_MAX: suhu udara maksimum harian 2 m (C).
T2M_MIN: suhu udara minimum harian 2 m (C).
RH2M: kelembapan relatif 2 m (%).
QV2M: kelembapan spesifik 2 m (g/kg).
T2MDEW: suhu titik embun/beku 2 m (C).
PS: tekanan udara permukaan (kPa).
WS2M, WS10M: kecepatan angin 2 m dan 10 m (m/s).
WD10M: arah angin 10 m (derajat).
ALLSKY_SFC_SW_DWN: radiasi gelombang pendek turun di permukaan, segala kondisi langit (MJ/m2/hari, community AG).
Satuan dan deskripsi asli API tersimpan dalam metadata.json masing-masing lokasi.

FILE DAN AGREGASI
Harian: nilai asli API, tanpa interpolasi; nilai missing -999 diubah menjadi sel kosong, bukan nol.
Bulanan: PRECTOTCORR dijumlahkan menjadi mm/bulan. Variabel lainnya dirata-ratakan dari nilai harian.
T2M_MAX_rata_rata adalah rata-rata maksimum harian, bukan maksimum absolut bulanan. T2M_MIN mengikuti aturan yang sama.
Radiasi bulanan adalah rata-rata energi harian (MJ/m2/hari), bukan total energi sebulan.
Arah angin memakai rata-rata sirkular tanpa bobot kecepatan; kosong jika arah saling meniadakan.
Agregat bulanan hanya diisi jika semua hari memiliki data valid untuk variabel tersebut. Kolom hari_valid menunjukkan kelengkapan.
kelengkapan_data.csv mencatat jumlah missing serta tanggal valid pertama/terakhir untuk setiap lokasi dan variabel.
Metadata dan JSON sumber asli tersedia untuk penelusuran. Batas awal data dapat berbeda antarvariabel.
CSV: UTF-8 BOM, pemisah koma, desimal titik, tanggal ISO.
Pemeriksaan: tanggal unik dan rentang benar, jumlah baris harian/bulanan, nilai numerik hingga, serta rekonsiliasi total hujan harian dan bulanan.
