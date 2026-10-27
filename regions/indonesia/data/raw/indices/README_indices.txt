INDEKS IKLIM UNTUK ANALISIS LOKASI INDONESIA, 1981-2025
540 bulan, Januari 1981 sampai Desember 2025.
Indeks bersifat regional/global, bukan pengukuran lokal tiap kota. Gabungkan ke data bulanan 20 lokasi Indonesia menggunakan kolom bulan (YYYY-MM). Nilai indeks bulan yang sama sama untuk semua lokasi.

FILE UTAMA: indeks_iklim_bulanan_1981_2025.csv
nino34_anom_C, nino4_anom_C: anomali SST bulanan (C), NOAA CPC ERSST v5, baseline 1991-2020. Diambil dari kolom ANOM untuk NINO3.4 dan NINO4; bukan SST absolut, bukan ONI tiga bulanan, bukan ERSSTv6.
dmi_hadisst_C: NOAA PSL DMI dari HadISST1.1 (C), Samudra Hindia barat minus tenggara. Data berlabel preliminary pada sumber; tidak dihitung ulang atau distandardisasi ulang.
soi_cru: indeks tekanan Tahiti-Darwin ternormalisasi CRU, tanpa satuan, presisi sumber 3 desimal. Jangan diasumsikan memakai skala SOI BoM yang dikalikan 10.
rmm1_mean, rmm2_mean: rata-rata aritmetika komponen RMM harian BoM, tanpa satuan. Hanya diisi jika seluruh hari bulan tersebut valid; jumlah hari valid/missing tersedia dalam file MJO bulanan.

MJO adalah osilasi intramusiman. Rata-rata bulanan RMM dapat saling meniadakan saat fase berubah; nilai kecil tidak otomatis berarti MJO tidak aktif. Data harian, fase, dan amplitudo disertakan untuk analisis lanjutan. Amplitudo rata-rata tidak dihitung sebagai akar kuadrat rata-rata RMM1 dan RMM2.
Catatan metode sumber BoM: sampai 2013-12-31 variasi SST1 (ENSO) dan rata-rata 120 hari dihilangkan; sejak 2014-01-01 hanya rata-rata 120 hari dihilangkan. Keterangan metode per hari dipertahankan dalam teks sumber.
Tautan lama BoM (rmm.txt) berhenti 2024-02-24 dan TIDAK digunakan. Analisis memakai rmm_current.txt dari endpoint clim_data/IDCKGEM000.

File masing-masing indeks tersedia terpisah. kelengkapan_indeks.csv dan metadata.json memuat hasil pemeriksaan, URL dan hash sumber. Data hilang dikosongkan, tidak diinterpolasi atau diganti nol.
CSV memakai UTF-8 BOM, koma sebagai delimiter, titik desimal, tanggal ISO.

SUMBER DAN ATRIBUSI
NOAA Climate Prediction Center: https://www.cpc.ncep.noaa.gov/data/indices/
NOAA Physical Sciences Laboratory: https://psl.noaa.gov/data/timeseries/month/DMI/
Climatic Research Unit, University of East Anglia: https://crudata.uea.ac.uk/cru/data/soi/
CRU SOI menggunakan metode Ropelewski & Jones (1987); lisensi Open Database License dan Database Contents License (attribution/share-alike), sesuai laman sumber.
Australian Bureau of Meteorology: https://www.bom.gov.au/clim_data/IDCKGEM000/rmm.74toRealtime.txt
URL unduhan persis terdapat dalam metadata.json. Simpan atribusi dan ikuti ketentuan penyedia saat mendistribusikan ulang.
