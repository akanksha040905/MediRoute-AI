"""
sample_data_generator.py — Realistic Karnataka hospital seed data (231 hospitals).
Used only when data/icu_beds.csv is missing entirely.
"""

import random
import pandas as pd
from datetime import datetime


def generate_sample_data() -> pd.DataFrame:
    random.seed(42)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Real-ish Karnataka hospital names across districts
    hospital_templates = [
        # Bengaluru
        ("Manipal Hospital Whitefield", 12.9716, 77.7499, "+91-080-25023456"),
        ("Fortis Hospital Bannerghatta", 12.8958, 77.5972, "+91-080-66214444"),
        ("Narayana Health City", 12.8992, 77.6101, "+91-080-71222222"),
        ("Apollo Hospital Jayanagar", 12.9249, 77.5933, "+91-080-26304050"),
        ("St. John's Medical College Hospital", 12.9352, 77.6245, "+91-080-22065000"),
        ("Vikram Hospital Millers Road", 12.9899, 77.5882, "+91-080-22262626"),
        ("BGS Gleneagles Global Hospital", 12.9141, 77.5488, "+91-080-26751000"),
        ("Columbia Asia Hospital Hebbal", 13.0358, 77.5970, "+91-080-61888888"),
        ("MS Ramaiah Memorial Hospital", 13.0197, 77.5570, "+91-080-23606789"),
        ("Sakra World Hospital", 12.9352, 77.6889, "+91-080-49694969"),
        ("Aster CMI Hospital", 13.0435, 77.5953, "+91-080-43422222"),
        ("NIMHANS", 12.9407, 77.5955, "+91-080-46110007"),
        ("Bowring & Lady Curzon Hospital", 12.9698, 77.6014, "+91-080-25544444"),
        ("Victoria Hospital", 12.9610, 77.5748, "+91-080-26700435"),
        ("Kempegowda Institute of Medical Sciences", 12.9340, 77.5534, "+91-080-26622900"),
        ("M.S. Ramaiah Teaching Hospital", 13.0252, 77.5503, "+91-080-23608888"),
        ("Sparsh Hospital", 13.0093, 77.5520, "+91-080-43030000"),
        ("Cloudnine Hospital Jayanagar", 12.9362, 77.5860, "+91-080-67676767"),
        ("Sagar Hospital Jayanagar", 12.9208, 77.5866, "+91-080-26931000"),
        ("Bangalore Baptist Hospital", 13.0086, 77.5599, "+91-080-22024700"),
        # Mysuru
        ("JSS Hospital Mysuru", 12.3052, 76.6551, "+91-0821-2548400"),
        ("K.R. Hospital Mysuru", 12.2961, 76.6394, "+91-0821-2423001"),
        ("Basappa Memorial Hospital Mysuru", 12.2994, 76.6400, "+91-0821-2421094"),
        ("Columbia Asia Hospital Mysuru", 12.2905, 76.6380, "+91-0821-6666888"),
        ("Apollo BGS Hospital Mysuru", 12.3196, 76.6351, "+91-0821-2560001"),
        # Hubli-Dharwad
        ("KIMS Hubli", 15.3522, 75.1244, "+91-0836-2376001"),
        ("District Hospital Dharwad", 15.4640, 75.0050, "+91-0836-2440228"),
        ("SDM Hospital Dharwad", 15.4551, 74.9967, "+91-0836-2447382"),
        ("Navodaya Medical College Raichur", 16.1921, 77.3666, "+91-08532-225533"),
        ("General Hospital Hubli", 15.3647, 75.1240, "+91-0836-2256800"),
        # Mangaluru
        ("KMC Hospital Mangaluru", 12.8754, 74.8754, "+91-0824-2445858"),
        ("Father Muller Medical College", 12.8870, 74.8454, "+91-0824-2238000"),
        ("AJ Hospital Mangaluru", 12.8924, 74.8460, "+91-0824-2225533"),
        ("Wenlock District Hospital", 12.8757, 74.8405, "+91-0824-2444255"),
        ("Kasturba Medical College Mangaluru", 12.8557, 74.8305, "+91-0824-2211111"),
        # Belagavi
        ("KLE Hospital Belagavi", 15.8497, 74.4977, "+91-0831-2470000"),
        ("District Hospital Belagavi", 15.8656, 74.5101, "+91-0831-2407777"),
        ("BIMS Belagavi", 15.8602, 74.5028, "+91-0831-2456001"),
        # Kalaburagi
        ("ESIC Hospital Kalaburagi", 17.3398, 76.8193, "+91-08472-263144"),
        ("District Hospital Kalaburagi", 17.3297, 76.8202, "+91-08472-261001"),
        ("Basaveshwar Teaching Hospital", 17.3380, 76.8149, "+91-08472-265200"),
        # Shivamogga
        ("McGann Teaching Hospital Shivamogga", 13.9299, 75.5681, "+91-08182-225177"),
        ("District Hospital Shivamogga", 13.9232, 75.5605, "+91-08182-222001"),
        # Davanagere
        ("SS Institute of Medical Sciences Davangere", 14.4686, 75.9202, "+91-08192-208888"),
        ("Chigateri General Hospital Davangere", 14.4667, 75.9199, "+91-08192-231001"),
        # Tumakuru
        ("Siddaganga Hospital Tumakuru", 13.3419, 77.1015, "+91-0816-2271999"),
        ("District Hospital Tumakuru", 13.3380, 77.1010, "+91-0816-2270601"),
        # Hassan
        ("Hassan Institute of Medical Sciences", 13.0072, 76.1013, "+91-08172-268011"),
        ("District Hospital Hassan", 13.0055, 76.0967, "+91-08172-262001"),
        # Ballari
        ("VIMS Ballari", 15.1394, 76.9214, "+91-08392-275101"),
        ("District Hospital Ballari", 15.1486, 76.9274, "+91-08392-271001"),
    ]

    # Expand to ~231 hospitals by adding district-level and taluk hospitals
    district_hospitals = [
        "Bidar District Hospital", "Yadgir District Hospital", "Raichur District Hospital",
        "Koppal District Hospital", "Gadag District Hospital", "Haveri District Hospital",
        "Uttara Kannada District Hospital", "Dakshina Kannada District Hospital",
        "Udupi District Hospital", "Chikkamagaluru District Hospital",
        "Kodagu District Hospital", "Mandya District Hospital",
        "Chamarajanagar District Hospital", "Chitradurga District Hospital",
        "Chikkaballapur District Hospital", "Kolar District Hospital",
        "Ramanagara District Hospital", "Bengaluru Rural District Hospital",
        "Vijayapura District Hospital", "Bagalkot District Hospital",
    ]

    district_coords = [
        (17.9148, 76.8217), (16.7665, 77.1333), (16.2083, 77.3591),
        (15.3554, 76.1547), (15.4313, 75.6323), (14.7939, 75.3986),
        (14.8134, 74.6313), (12.8406, 74.8927), (13.3301, 74.7470),
        (13.3161, 75.7720), (12.4244, 75.7382), (12.5244, 76.8962),
        (11.9432, 77.0060), (14.2251, 76.4015), (13.4335, 77.7263),
        (13.1357, 78.1291), (12.7160, 77.2822), (13.1480, 77.5068),
        (16.8302, 75.7100), (16.1791, 75.6976),
    ]

    district_contacts = [
        "+91-08482-225001", "+91-08473-252001", "+91-08532-220001",
        "+91-08539-220001", "+91-08372-230001", "+91-08375-220001",
        "+91-08382-222001", "+91-0824-2440001", "+91-0820-2520001",
        "+91-08262-220001", "+91-08272-225001", "+91-08232-222001",
        "+91-08226-222001", "+91-08194-222001", "+91-08156-222001",
        "+91-08152-222001", "+91-08027-270001", "+91-080-23331001",
        "+91-08352-250001", "+91-08354-220001",
    ]

    rows = []
    for (name, lat, lon, contact) in hospital_templates:
        total = random.randint(40, 200)
        avail = random.randint(2, max(3, total // 4))
        rows.append({
            "hospital": name, "contact_number": contact,
            "latitude": lat, "longitude": lon,
            "total_beds": total, "available_beds": avail,
            "avg_daily_patients": random.randint(30, total - 5),
            "critical_patients": random.randint(2, 25),
            "ventilators": random.randint(5, 50),
            "waiting_queue": random.randint(0, 20),
            "icu_specialist_count": random.randint(3, 18),
            "oxygen_supply_pct": round(random.uniform(65, 99), 1),
            "last_updated": now,
        })

    for i, (name, (lat, lon), contact) in enumerate(
        zip(district_hospitals, district_coords, district_contacts)
    ):
        total = random.randint(30, 120)
        avail = random.randint(1, max(2, total // 5))
        rows.append({
            "hospital": name, "contact_number": contact,
            "latitude": lat, "longitude": lon,
            "total_beds": total, "available_beds": avail,
            "avg_daily_patients": random.randint(20, total - 5),
            "critical_patients": random.randint(1, 15),
            "ventilators": random.randint(2, 20),
            "waiting_queue": random.randint(0, 15),
            "icu_specialist_count": random.randint(2, 10),
            "oxygen_supply_pct": round(random.uniform(60, 95), 1),
            "last_updated": now,
        })

    # Pad to 231 with taluk-level hospitals scattered across Karnataka
    taluk_names = [
        "Taluk General Hospital Sira", "Taluk Hospital Tiptur", "Taluk Hospital Kunigal",
        "Taluk Hospital Madhugiri", "Taluk Hospital Pavagada", "Taluk Hospital Gauribidanur",
        "Taluk Hospital Chintamani", "Taluk Hospital Srinivasapur", "Taluk Hospital Mulbagal",
        "Taluk Hospital Bangarpet", "Taluk Hospital Robertsonpet", "Taluk Hospital Malur",
        "CHC Anekal", "CHC Hoskote", "CHC Doddaballapura",
        "CHC Nelamangala", "CHC Magadi", "CHC Kanakapura",
        "CHC Channapatna", "CHC Maddur", "CHC Mandya",
        "CHC Malavalli", "CHC Nagamangala", "CHC Krishnarajapet",
        "CHC Pandavapura", "CHC Srirangapatna", "CHC T. Narsipur",
        "CHC Hunsur", "CHC H.D. Kote", "CHC Nanjangud",
        "CHC Kollegal", "CHC Chamarajanagar", "CHC Gundlupet",
        "CHC Yelandur", "CHC Mysuru Rural", "CHC Periyapatna",
        "CHC K.R. Nagar", "CHC Heggadadevankote", "CHC Piriyapatna",
        "CHC Arsikere", "CHC Belur", "CHC Alur",
        "CHC Sakleshpur", "CHC Mudigere", "CHC Kadur",
        "CHC Tarikere", "CHC Birur", "CHC Jagalur",
        "CHC Harapanahalli", "CHC Honnali", "CHC Channagiri",
        "CHC Nyamathi", "CHC Bhadravathi", "CHC Thirthahalli",
        "CHC Hosanagara", "CHC Sagar", "CHC Soraba",
        "CHC Shikaripura", "CHC Shiralakoppa", "CHC Kundapura",
        "CHC Karkala", "CHC Udupi", "CHC Brahmavar",
        "CHC Hebri", "CHC Byndoor", "CHC Shirva",
        "CHC Belthangady", "CHC Puttur", "CHC Sullia",
        "CHC Bantwal", "CHC Moodabidri", "CHC Vitla",
        "CHC Sirsi", "CHC Siddapur", "CHC Yellapur",
        "CHC Kumta", "CHC Ankola", "CHC Karwar",
        "CHC Honavar", "CHC Bhatkal", "CHC Haliyal",
        "CHC Dharwad Rural", "CHC Navalgund", "CHC Kundgol",
        "CHC Kalghatgi", "CHC Ron", "CHC Nargund",
        "CHC Shirahatti", "CHC Gadag Rural", "CHC Mundargi",
        "CHC Lakshmeshwar", "CHC Savanur", "CHC Byadgi",
        "CHC Ranebennur", "CHC Hangal", "CHC Shiggaon",
        "CHC Hirekerur", "CHC Rona", "CHC Kalaghatagi",
        "CHC Dandeli", "CHC Mundgod", "CHC Joida",
        "CHC Khanapur", "CHC Bailhongal", "CHC Ramdurg",
        "CHC Gokak", "CHC Chikkodi", "CHC Hukkeri",
        "CHC Athani", "CHC Kagwad", "CHC Raibag",
        "CHC Soundatti", "CHC Mudhol", "CHC Jamkhandi",
        "CHC Bilgi", "CHC Badami", "CHC Hungund",
        "CHC Ilkal", "CHC Bagalkot Rural", "CHC Rabkavi-Banhatti",
        "CHC Sindgi", "CHC Indi", "CHC Muddebihal",
        "CHC Basavana Bagewadi", "CHC Bijapur Rural", "CHC Chadchan",
        "CHC Aurad", "CHC Bhalki", "CHC Humnabad",
        "CHC Basavakalyan", "CHC Bidar Rural", "CHC Kamalapur",
        "CHC Shahapur", "CHC Shorapur", "CHC Gurmatkal",
        "CHC Yadgir Rural", "CHC Raichur Rural", "CHC Manvi",
        "CHC Sindhanur", "CHC Devadurga", "CHC Lingasugur",
        "CHC Gangavathi", "CHC Yelburga", "CHC Kustagi",
        "CHC Koppal Rural", "CHC Bellary Rural", "CHC Sandur",
        "CHC Hadagali", "CHC Hagaribommanahalli", "CHC Siruguppa",
        "CHC Kudligi", "CHC Hosapete", "CHC Kampli",
        "CHC Challakere", "CHC Holalkere", "CHC Chitradurga Rural",
        "CHC Hiriyur", "CHC Molakalmuru", "CHC Pavagada",
        "CHC Chikkanayakanahalli", "CHC Tiptur", "CHC Gubbi",
        "CHC Tumakuru Rural", "CHC Koratagere", "CHC Madhugiri",
    ]

    # Karnataka lat/lon bounding box
    kar_lats = [11.8, 18.0]
    kar_lons = [74.0, 78.5]

    for name in taluk_names:
        if len(rows) >= 231:
            break
        total = random.randint(20, 80)
        avail = random.randint(0, max(1, total // 6))
        lat = round(random.uniform(*kar_lats), 4)
        lon = round(random.uniform(*kar_lons), 4)
        dist_code = random.randint(10000, 99999)
        rows.append({
            "hospital": name, "contact_number": f"+91-0{random.randint(8100,8999)}-{dist_code}",
            "latitude": lat, "longitude": lon,
            "total_beds": total, "available_beds": avail,
            "avg_daily_patients": random.randint(10, total - 2),
            "critical_patients": random.randint(0, 10),
            "ventilators": random.randint(0, 10),
            "waiting_queue": random.randint(0, 12),
            "icu_specialist_count": random.randint(1, 6),
            "oxygen_supply_pct": round(random.uniform(55, 95), 1),
            "last_updated": now,
        })

    return pd.DataFrame(rows[:231])
