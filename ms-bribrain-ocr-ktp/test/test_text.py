import pytest
from utils.usecase import (matching_agama, matching_alamat, matching_nik, matching_jeniskelamin, 
                          matching_rtrw, matching_status, matching_keldesa, matching_kecamatan,
                          matching_tempatlahir, matching_nama, matching_tempatlahir_new)


# --------------------------
# Test matching_nik
# --------------------------
@pytest.mark.parametrize(
    "data,expected",
    [
        (['：3578242708960003', 0.9904646277427673, [[288, 92], [673, 90], [673, 121], [288, 123]]], "3578242708960003"),  # valid
        (['：3578242708960o03', 0.9904646277427673, [[288, 92], [673, 90], [673, 121], [288, 123]]], "3578242708960003"),  # valid
        (['：3578242708960003', 0.6904646277427673, [[288, 92], [673, 90], [673, 121], [288, 123]]], ""),  # valid
        (['：3578242708960', 0.9904646277427673, [[288, 92], [673, 90], [673, 121], [288, 123]]], ""),  # valid
        (['：357824270896000312', 0.9904646277427673, [[288, 92], [673, 90], [673, 121], [288, 123]]], ""),  # valid
        (['', 0.9904646277427673, [[288, 92], [673, 90], [673, 121], [288, 123]]], ""),  # valid
        # ((None, 0.9904646277427673, [[288, 92], [673, 90], [673, 121], [288, 123]]], ""),  # valid
    ]
)
def test_matching_nik(data, expected):
    assert matching_nik(data) == expected


# --------------------------
# Test matching_alamat
# --------------------------
@pytest.mark.parametrize(
    "data,next_data,expected",
    [
        (
            ('Alamat', 0.9807361960411072, [[135, 212], [204, 212], [204, 234], [135, 234]]), 
            ('JL.Merdeka N20', 0.9515178203582764, [[295, 212], [457, 212], [457, 233], [295, 233]]), 
            "JL.MERDEKA N20"
         ),  # valid
        (
            ('Alamat', 0.9807361960411072, [[135, 212], [204, 212], [204, 234], [135, 234]]), 
            ('JL.Merdeka N20', 0.4515178203582764, [[295, 212], [457, 212], [457, 233], [295, 233]]), 
            ""
         ),  # valid
        (
            ('Alamat', 0.9807361960411072, [[135, 212], [204, 212], [204, 234], [135, 234]]), 
            (':JL.Merdeka N20', 0.9515178203582764, [[295, 212], [457, 212], [457, 233], [295, 233]]), 
            "JL.MERDEKA N20"
        ),  # valid
    ]
)
def test_matching_alamat(data, next_data, expected):
    assert matching_alamat(key="alamat", data=data, next_data=next_data) == expected

# --------------------------
# Test matching_agama
# --------------------------
@pytest.mark.parametrize(
    "key,data,next_data,expected",
    [
        ("agama", ('Agama', 0.9888507723808289,[[130, 302], [201, 305], [200, 331], [129, 328]]), 
         ('ISLAM', 0.9912465214729309, [[309, 304], [376, 304], [376, 326], [309, 326]]), 
         "ISLAM"),  # valid
        ("islam", ('ISLAM', 0.9912465214729309, [[309, 304], [376, 304], [376, 326], [309, 326]]), 
         ('Agama', 0.9888507723808289,[[130, 302], [201, 305], [200, 331], [129, 328]]), 
         "ISLAM"),  # valid
        ("agama", ('Agama', 0.9888507723808289,[[130, 302], [201, 305], [200, 331], [129, 328]]), 
         ('ISLA', 0.9912465214729309, [[309, 304], [376, 304], [376, 326], [309, 326]]), 
         "ISLAM"),  # valid
        ("agama", ('Agama', 0.9888507723808289,[[130, 302], [201, 305], [200, 331], [129, 328]]), 
         ('ISLAa', 0.9912465214729309, [[309, 304], [376, 304], [376, 326], [309, 326]]), 
         "ISLAM"),  # valid
        ("agama", ('Agama', 0.9888507723808289,[[130, 302], [201, 305], [200, 331], [129, 328]]), 
         ('TSL4M', 0.9912465214729309, [[309, 304], [376, 304], [376, 326], [309, 326]]), 
         ""),  # valid
    ]
)
def test_matching_agama(key, data, next_data, expected):
    assert matching_agama(key=key, data=data, next_data=next_data) == expected

# --------------------------
# Test matching_jeniskelamin
# --------------------------
@pytest.mark.parametrize(
    "key,data,next_data,expected",
    [
        (
            "jenis kelamin",
            ('Jenis kelamin', 0.9504479169845581, [[136, 189], [259, 189], [259, 210], [136, 210]]), 
            (':LAKI-LAKI', 0.88450688123703, [[301, 188], [409, 188], [409, 209], [301, 209]]), 
            "LAKI-LAKI"
         ),
        (
            "jenis kelamin",
            ('Jenis kelamin', 0.9504479169845581, [[136, 189], [259, 189], [259, 210], [136, 210]]), 
            ('AKI-LAKI', 0.88450688123703, [[301, 188], [409, 188], [409, 209], [301, 209]]), 
            "LAKI-LAKI"
         ),
        (
            "jenis kelamin",
            ('Jenis kelamin', 0.9504479169845581, [[136, 189], [259, 189], [259, 210], [136, 210]]), 
            ('AKI-AKI', 0.88450688123703, [[301, 188], [409, 188], [409, 209], [301, 209]]), 
            "LAKI-LAKI"
         ),
        (
            "jenis kelamin",
            ('Jenis kelamin', 0.9504479169845581, [[136, 189], [259, 189], [259, 210], [136, 210]]), 
            (':LAKI-LAKI', 0.48450688123703, [[301, 188], [409, 188], [409, 209], [301, 209]]), 
            ""
         ),
        (
            "laki-laki",
            (':LAKI-LAKI', 0.9504479169845581, [[136, 189], [259, 189], [259, 210], [136, 210]]), 
            ('Gol.Darah', 0.48450688123703, [[301, 188], [409, 188], [409, 209], [301, 209]]), 
            "LAKI-LAKI"
         ),
    ]
)
def test_matching_jeniskelamin(key,data, next_data, expected):
    assert matching_jeniskelamin(key=key, data=data, next_data=next_data) == expected

# --------------------------
# Test rt/rw
# --------------------------
@pytest.mark.parametrize(
    "next_data,expected",
    [
        # ✅ Case normal, ada RT/RW dengan separator "/"
        (
            ('：002/005', 0.938106894493103, [[301, 235], [391, 235], [391, 256], [301, 256]]),
            ("002", "005")
         ),
        (
            ('002005', 0.938106894493103, [[301, 235], [391, 235], [391, 256], [301, 256]]),
            ("002", "005")
         ),
        (
            ('00205', 0.938106894493103, [[301, 235], [391, 235], [391, 256], [301, 256]]),
            ("002", "005")
         ),
        (
            ('：002/005', 0.538106894493103, [[301, 235], [391, 235], [391, 256], [301, 256]]),
            ("", "")
         ),
    ]
)
def test_matching_rtrw(next_data, expected):
    assert matching_rtrw(next_data=next_data) == expected

# --------------------------
# Test status_perkawinan
# --------------------------
@pytest.mark.parametrize(
    "key,data,next_data,expected",
    [
        # ✅ Ambil dari next_data karena key cocok dan panjang < 20
        (
            "status perkawinan",
            ('Status Perkawinart', 0.9330787658691406, [[133, 329], [311, 329], [311, 350], [133, 350]]),
            ('BELUM KAWIN', 0.9751880764961243, [[308, 328], [456, 328], [456, 349], [308, 349]]),
            "BELUM KAWIN"
        ),
        (
            "belum kawin",
            ('BELM KAWIN', 0.9751880764961243, [[308, 328], [456, 328], [456, 349], [308, 349]]),
            ('Status Perkawinart', 0.9330787658691406, [[133, 329], [311, 329], [311, 350], [133, 350]]),
            "BELUM KAWIN"
        ),
        (
            "kawin",
            ('KWN', 0.9751880764961243, [[308, 328], [456, 328], [456, 349], [308, 349]]),
            ('Status Perkawinart', 0.9330787658691406, [[133, 329], [311, 329], [311, 350], [133, 350]]),
            ""
        ),
        (
            "status perkawinan",
            ('Status Perkawinart', 0.9330787658691406, [[133, 329], [311, 329], [311, 350], [133, 350]]),
            ('CERA1 MAT1', 0.9751880764961243, [[308, 328], [456, 328], [456, 349], [308, 349]]),
            "CERAI MATI"
        ),
        (
            "status perkawinan",
            ('Status Perkawinart', 0.9330787658691406, [[133, 329], [311, 329], [311, 350], [133, 350]]),
            ('BELM KAW', 0.9751880764961243, [[308, 328], [456, 328], [456, 349], [308, 349]]),
            "BELUM KAWIN"
        ),
    ]
)
def test_matching_status(key,data, next_data, expected):
    assert matching_status(key=key, data=data, next_data=next_data) == expected

# --------------------------
# Test kel_desa
# --------------------------
@pytest.mark.parametrize(
    "data,next_data,expected",
    [
        # ✅ next_data valid → return uppercase
        (
            ('Kel/Desa', 0.966785192489624, [[181, 257], [270, 257], [270, 281], [181, 281]]),
            ('Ngageljo', 0.9615257978439331, [[311, 258], [457, 258], [457, 279], [311, 279]]),
            "NGAGELJO"
        ),
        (
            ('Kel/Desa', 0.966785192489624, [[181, 257], [270, 257], [270, 281], [181, 281]]),
            ('Ngageljo', 0.6615257978439331, [[311, 258], [457, 258], [457, 279], [311, 279]]),
            ""
        ),
    ]
)
def test_matching_keldesa(data, next_data, expected):
    assert matching_keldesa(key="kel/desa", data=data, next_data=next_data) == expected

# --------------------------
# Test matching_kecamatan
# --------------------------
@pytest.mark.parametrize(
    "data,next_data,expected",
    [
        # ✅ Ambil dari data, ada huruf kapital sesuai regex
        (
            ('Kecamatan', 0.9887179136276245,[[182, 281], [289, 281], [289, 302], [182, 302]]),
            ('WONOKROMO', 0.9488624930381775, [[311, 282], [456, 282], [456, 303], [311, 303]]),
            "WONOKROMO"
         ),
        (
            ('Kecamatan : WONOKROMO', 0.9887179136276245,[[182, 281], [289, 281], [289, 302], [182, 302]]),
            ('awwww', 0.9488624930381775, [[311, 282], [456, 282], [456, 303], [311, 303]]),
            "WONOKROMO"
         ),
        (
            ('Kecamatan', 0.9887179136276245,[[182, 281], [289, 281], [289, 302], [182, 302]]),
            ('WONOKROMO', 0.5488624930381775, [[311, 282], [456, 282], [456, 303], [311, 303]]),
            ""
         ),
    ]
)
def test_matching_kecamatan(data, next_data, expected):
    assert matching_kecamatan(key="kecamatan", data=data, next_data=next_data) == expected

# --------------------------
# Test matching_tempatlahir
# --------------------------
@pytest.mark.parametrize(
    "data,next_data,expected",
    [
        # ✅ Ambil dari data (digit >= 6)
        (
            ['Tempat/Tgl Lahir', 0.9260899424552917, [[138, 166], [305, 166], [305, 187], [138, 187]]], 
            ['SURABAYA,27-08-1996', 0.9624438881874084, [[295, 166], [538, 164], [538, 185], [295, 187]]], 
            ("SURABAYA", "27 Aug 1996")
         ),

        # ✅ Ambil dari next_data (digit >= 6)
        (
            ['Tempat/Tgl Lahir', 0.9260899424552917, [[138, 166], [305, 166], [305, 187], [138, 187]]], 
            ['SURABAYA 2708-1996', 0.9624438881874084, [[295, 166], [538, 164], [538, 185], [295, 187]]], 
            ("SURABAYA", "27 Aug 1996")
         ),

        # ✅ Fallback kalau ada 'jenis' di next_data
        (
            ['Tempat/Tgl Lahir', 0.9260899424552917, [[138, 166], [305, 166], [305, 187], [138, 187]]], 
            ['SURABAYA 27081996', 0.9624438881874084, [[295, 166], [538, 164], [538, 185], [295, 187]]], 
            ("SURABAYA", "27 Aug 1996")
        ),

        # ✅ Fallback tanpa 'jenis'
        (
            ['Tempat/Tgl Lahir', 0.9260899424552917, [[138, 166], [305, 166], [305, 187], [138, 187]]], 
            ['SURABAYA 2708-1996', 0.9624438881874084, [[295, 166], [538, 164], [538, 185], [295, 187]]], 
            ("SURABAYA", "27 Aug 1996")
         ),

        # ✅ Normalisasi tempat panjang (ambil huruf kapital aja)
        (
            ['Tempat/Tgl Lahir', 0.9260899424552917, [[138, 166], [305, 166], [305, 187], [138, 187]]], 
            ['SURABAYA 2708-1996', 0.9624438881874084, [[295, 166], [538, 164], [538, 185], [295, 187]]], 
            ("SURABAYA", "27 Aug 1996")
         ),
        (
            ['Tempat/Tgl Lahir', 0.9260899424552917, [[138, 166], [305, 166], [305, 187], [138, 187]]], 
            ['TORAJA.29-01-1978', 0.9624438881874084, [[295, 166], [538, 164], [538, 185], [295, 187]]], 
            ("TORAJA", "29 Jan 1978")
         ),

        # ❌ Data dan next_data tidak valid → return kosong
        (
            ['Tempat/Tgl Lahir', 0.9260899424552917, [[138, 166], [305, 166], [305, 187], [138, 187]]], 
            ['SURABAYA 278-1996', 0.9624438881874084, [[295, 166], [538, 164], [538, 185], [295, 187]]], 
            ("", "")
         ),
    ]
)
def test_matching_tempatlahir(data, next_data, expected):
    tempat, tgl = matching_tempatlahir(data, next_data)
    assert (tempat, tgl) == expected

# --------------------------
# Test matching_tempatlahir_new
# --------------------------
@pytest.mark.parametrize(
    "data,next_data,expected",
    [
        # ✅ Ambil dari data (digit >= 6)
        (
            ['Tempat/Tgl Lahir', 0.9260899424552917, [[138, 166], [305, 166], [305, 187], [138, 187]]], 
            ['SURABAYA,27-08-1996', 0.9624438881874084, [[295, 166], [538, 164], [538, 185], [295, 187]]], 
            ("SURABAYA", "27 Aug 1996")
         ),

        # ✅ Ambil dari next_data (digit >= 6)
        (
            ['Tempat/Tgl Lahir', 0.9260899424552917, [[138, 166], [305, 166], [305, 187], [138, 187]]], 
            ['SURABAYA 2808-1996', 0.9624438881874084, [[295, 166], [538, 164], [538, 185], [295, 187]]], 
            ("SURABAYA", "28 Aug 1996")
         ),

        # ✅ Fallback kalau ada 'jenis' di next_data
        (
            ['Tempat/Tgl Lahir', 0.9260899424552917, [[138, 166], [305, 166], [305, 187], [138, 187]]], 
            ['SURABAYA 29081996', 0.9624438881874084, [[295, 166], [538, 164], [538, 185], [295, 187]]], 
            ("SURABAYA", "29 Aug 1996")
        ),

        # ✅ Fallback tanpa 'jenis'
        (
            ['Tempat/Tgl Lahir', 0.9260899424552917, [[138, 166], [305, 166], [305, 187], [138, 187]]], 
            ['SURABAYA 2708-1996', 0.9624438881874084, [[295, 166], [538, 164], [538, 185], [295, 187]]], 
            ("SURABAYA", "27 Aug 1996")
         ),
        (
            ['Tempat/Tgl Lahir', 0.9260899424552917, [[138, 166], [305, 166], [305, 187], [138, 187]]], 
            ['PSURABAYA,27C8-1996', 0.9624438881874084, [[295, 166], [538, 164], [538, 185], [295, 187]]], 
            ("PSURABAYA", "27 Aug 1996")
         ),
    ]
)
def test_matching_tempatlahir_new(data, next_data, expected):
    tempat, tgl = matching_tempatlahir_new(data, next_data)
    assert (tempat, tgl) == expected

# --------------------------
# Test matching_nama
# --------------------------
@pytest.mark.parametrize(
    "data,expected",
    [
        # ✅ Nama valid → uppercase
        (('FACHRUL HIDAYAT', 0.9479934573173523, [[302, 141], [500, 141], [500, 162], [302, 162]]), "FACHRUL HIDAYAT"),

        # ✅ Nama dengan karakter spesial
        (('!FACHRUL HIDAYAT', 0.9479934573173523, [[302, 141], [500, 141], [500, 162], [302, 162]]), "FACHRUL HIDAYAT"),

        # ✅ Nama dengan spasi ganda
        ((' FACHRUL HIDAYAT ', 0.9479934573173523, [[302, 141], [500, 141], [500, 162], [302, 162]]), "FACHRUL HIDAYAT"),

        # ❌ Confidence rendah → return kosong
        ((':FACHRUL HIDAYAT', 0.6479934573173523, [[302, 141], [500, 141], [500, 162], [302, 162]]), ""),

        # ❌ Data kosong
        (('', 0.9479934573173523, [[302, 141], [500, 141], [500, 162], [302, 162]]), ""),

        # ❌ None
        # ((None, 0.9479934573173523, [[302, 141], [500, 141], [500, 162], [302, 162]]], ""),
    ]
)
def test_matching_nama(data, expected):
    assert matching_nama(data=data) == expected
    