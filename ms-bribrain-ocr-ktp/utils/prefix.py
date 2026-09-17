keydata = (
    "nik", "nama", "tempat/tgl lahir", "jenis kelamin", "alamat", "rt/rw", "kel/desa", "kecamatan", "agama",
    "status perkawinan", "perempuan", "laki-laki", "belum kawin", "kawin", "cerai hidup", "cerai mati", "married",
    "islam", "kristen", "katholik", "hindu", "buddha", "konghucu", "christian"
)
keyid = ("nik", "nama", "tempat/tgl lahir", "jenis kelamin", "alamat", "rt/rw", "kel/desa", "kecamatan", "agama","status perkawinan")
# result = {"nik":["",0], "nama":["",0], "Tempat/Tgl Lahir":["",0], "jenis kelamin":["",0], "alamat":["",0], "rt/rw":["",0], "kel/desa":["",0], "kecamatan":["",0], "agama":["",0], "status perkawinan":["",0]}

mapping_pekerjaan= (
        "PELAJAR/MAHASISWA", "KARYAWAN SWASTA", "PNS", "TNI", "POLRI",
        "PETANI/PEKEBUN", "NELAYAN", "WIRASWASTA", "GURU", "DOKTER",
        "PERAWAT", "PENSIUNAN", "IBU RUMAH TANGGA", "BURUH", "SOPIR",
        "PEDAGANG", "NOTARIS", "PENGACARA", "DOSEN", "USTADZ", "PENDIDIK",
        "WARTAWAN", "JURNALIS", "KARYAWAN BUMN", "KARYAWAN BUMD", "KARYAWAN HONORER",
        "KARYAWAN", "OTHERS", "PERDAGANGAN", "MENGURUS RUMAH TANGGA", "PENSIUNAN",
        "BELUM/TIDAK BEKERJA", "SWASTA", "PEGAWAI SWASTA", "KARYAWAN SWASTA",
        )

mapping_status = ("BELUM KAWIN", "KAWIN", "CERAI HIDUP", "CERAI MATI", "MARRIED")
mapping_agama = ("ISLAM", "KRISTEN", "KATHOLIK", "HINDU", "BUDDHA", "KONGHUCU","CHRISTIAN")
mapping_jeniskelamin = ("LAKI-LAKI", "PEREMPUAN")

bulan_dict = {
        "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May",
        "06": "Jun", "07": "Jul", "08": "Aug", "09": "Sep",
        "10": "Oct", "11": "Nov", "12": "Dec"
    }

alphabet_mapping = {
        "O": "0",
        "o": "0",
        "A": "4",
        "q": "9",
        "l": "1",
        "z": "2",
        "Z": "2",
        "I": "1",
        "B": "8",
        "S": "5",
        "s": "5",
        "G": "6",
        "g": "9",
        "E": "3",
        "e": "3",
        "b": "6",
        "P": "9",
        "D": "0",
        "Q": "9",
        "C": "0",
        "c": "0",
        "L": "1",
        "T": "7",
    }

digit_mapping = {
    "0": "O",
    "4": "A",
    "9": "P",
    "1": "I",
    "2": "Z",
    "8": "B",
    "5": "S",
    "6": "G",
    "3": "e",
    "7": "T"
    }

symbol_mapping = {
        "+": "7",
        "[": "1",
        "]": "1",
        "!": "1",
        "|": "1",
        "/": "1",
        "\\": "1",
        "&": "8",
        "?": "7",
    }

keydata_repeat = ("rt/rw", "kel/desa", "kecamatan", "tempat/tgl lahir", "status perkawinan", "belum kawin", "kawin", "cerai hidup", "cerai mati", "married")
keydata_status = ("status perkawinan", "belum kawin", "kawin", "cerai hidup", "cerai mati", "married")
keydata_agama = ("agama", "ISLAM", "KRISTEN", "KATHOLIK", "HINDU", "BUDDHA", "KONGHUCU","CHRISTIAN")
keydata_jeniskelamin = ("jenis kelamin", "perempuan", "laki-laki")                            

data_key_unittest = ("nik", "jenis_kelamin", "nama", "tempat_lahir", "tanggal_lahir", "alamat", "rt", "rw", "kel_desa", "kecamatan", "agama", "status_perkawinan")

message_glare = "glare"
message_blur = "blur"
message_rotated = "tidak sejajar"
message_median = "kualitas buruk"
