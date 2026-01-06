import streamlit as st
import io
import copy
import random
from typing import Dict, List, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================
st.set_page_config(page_title="Turkish Election Simulator", layout="wide")

PARTIES = [
    "AK PARTI", "CHP", "MHP", "IYI PARTI", "DEM PARTI",
    "YENIDEN REFAH", "ZAFER", "TIP", "BBP", "UNDECIDED",
]

PARTIES_2023 = [p for p in PARTIES if p != "UNDECIDED"]
THRESHOLD_PERCENT = 7.0

# =============================================================================
# EMBEDDED DATA (ASCII ONLY)
# =============================================================================
DATA_2023_CSV = """Sira No,Il Adi,Milletvekili sayisi,Toplam Gecerli Oy,BBP,AK PARTI,YENIDEN REFAH,MHP,CHP,IYI PARTI,ZAFER PARTISI,DEM,TIP
1,ADANA,15,1378331,12594,421572,24556,149861,398645,149935,29480,133770,31347
2,ADIYAMAN,5,314939,3119,165550,15169,12754,58397,0,1955,38042,0
3,AFYONKARAHISAR,6,471248,5145,209089,14013,75657,88380,59015,10604,1507,1020
4,AGRI,4,212257,605,52483,4186,4805,22771,3466,0,118684,0
5,AKSARAY,4,250917,5304,110621,6465,52466,2,63281,3379,1477,628
6,AMASYA,3,230956,1914,91571,4133,38680,68984,16590,3486,780,845
7,ANKARA-1,13,1361329,11737,332736,26453,109393,575170,179283,41141,53827,0
8,ANKARA-2,11,1149060,16194,481080,40204,136526,237934,133811,40189,25120,13071
9,ANKARA-3,12,1331171,21035,425880,36563,142189,374117,186564,49077,32799,32647
10,ANTALYA,17,1661284,10993,476582,20738,163915,543903,190186,47322,78461,76491
11,ARDAHAN,2,53898,186,18680,532,1594,16166,2619,570,11723,0
12,ARTVIN,2,114073,438,42254,2701,10857,34508,15069,1344,981,1716
13,AYDIN,8,768239,6546,217092,5570,64983,277706,100533,17282,55478,0
14,BALIKESIR,9,881676,6069,303728,18373,70651,281637,130339,18326,17642,10672
15,BARTIN,2,131686,595,48764,2324,30933,41681,0,1911,419,0
16,BATMAN,5,325582,995,93243,3448,3675,25771,0,0,193025,0
17,BAYBURT,1,49046,360,29498,1452,8122,0,7980,667,207,96
18,BILECIK,2,149866,1291,55730,4911,15251,40413,18467,3382,3954,1242
19,BINGOL,3,144717,783,57100,14562,19401,7549,2818,0,35315,0
20,BITLIS,3,173436,561,64471,3635,6161,0,22914,0,70825,0
21,BOLU,3,210825,1594,78265,6851,49262,46165,15782,4063,1717,1142
22,BURDUR,3,178884,1243,65267,2157,24960,56174,20640,2575,1105,0
23,BURSA-1,10,1092375,9001,390460,34293,85837,299411,150123,41943,33529,14607
24,BURSA-2,10,1013312,6950,430594,41215,91991,217538,103378,36083,56557,0
25,CANAKKALE,4,390896,1700,123899,3781,25103,140054,64766,7860,7480,4820
26,CANKIRI,2,123191,1245,52805,3696,38322,0,22241,1895,367,297
27,CORUM,4,354060,2747,143260,11675,71358,109716,0,5398,1419,1255
28,DENIZLI,7,709052,10235,239237,9563,58729,229403,101081,18494,16716,7148
29,DIYARBAKIR,12,896596,1939,201583,13904,10207,69166,19864,0,561265,0
30,DUZCE,3,265570,2404,132477,10319,42876,59971,0,5754,2549,1151
31,EDIRNE,4,281477,659,65453,1857,21085,113225,58391,4241,4997,2936
32,ELAZIG,5,356398,1396,144077,13031,40446,73512,8390,1950,25066,0
33,ERZINCAN,2,151143,550,58403,2178,29787,54883,0,1460,1553,366
34,ERZURUM,6,429394,7526,186788,21410,71509,28119,41236,10282,42462,0
35,ESKISEHIR,6,614299,6793,201830,8717,43670,212328,86097,18617,12020,9924
36,GAZIANTEP,14,1133176,7828,510424,46838,108261,229703,61542,37425,102986,0
37,GIRESUN,4,299850,1754,129820,7421,46737,62141,38598,4623,721,1308
38,GUMUSHANE,2,78192,645,37469,2092,20640,2,14601,860,231,207
39,HAKKARI,3,148685,445,29435,2931,3988,9917,0,0,95302,0
40,HATAY,11,864726,7743,291634,15837,104557,247436,70101,11507,26088,75658
41,IGDIR,2,101639,205,35498,457,4581,6163,6706,0,45388,0
42,ISPARTA,4,285725,1536,93098,5196,58423,63979,47409,5552,1975,900
43,ISTANBUL-1,35,3769992,37850,1298399,112619,225296,1203625,295449,102722,234883,175881
44,ISTANBUL-2,27,2749298,19403,1049912,106588,153043,733861,237942,78649,202700,101991
45,ISTANBUL-3,36,3556584,24697,1266594,111920,228023,940947,288129,103850,380027,132485
46,IZMIR-1,14,1486087,7034,370269,12163,68652,632972,178748,34721,134381,0
47,IZMIR-2,14,1553179,6892,379331,14376,86707,631582,171799,36956,92075,85103
48,KAHRAMANMARAS,8,627604,9884,301738,34455,101159,100984,44594,10810,7264,0
49,KARABUK,3,158127,673,59693,13201,25158,34783,15438,3324,632,517
50,KARAMAN,3,163674,4394,67535,3715,29609,31522,18005,3081,803,444
51,KARS,3,141217,847,35723,2757,13494,23170,21327,0,40227,0
52,KASTAMONU,3,251359,2276,114999,10232,37593,54716,19289,4432,543,0
53,KAYSERI,10,915750,17477,374654,32291,174332,156702,92555,29838,7600,3543
54,KIRIKKALE,3,176696,1163,63166,3214,37436,47314,18074,2718,528,0
55,KIRKLARELI,3,260359,1299,64191,1701,12022,120779,40142,4059,4435,4566
56,KIRSEHIR,2,148589,1475,57729,2372,21997,43773,10797,2900,3677,589
57,KILIS,2,84073,364,33552,1968,22604,16900,4861,1598,724,0
58,KOCAELI,14,1359568,15287,538958,79279,112123,327083,130522,43088,77057,0
59,KONYA,15,1415892,28814,683672,72931,202108,190091,123996,42301,36133,4565
60,KUTAHYA,5,385562,4564,180937,27636,49940,64304,40917,6648,1031,969
61,MALATYA,6,425215,5738,192986,39912,54607,91558,16664,4805,12685,0
62,MANISA,10,987647,13135,312134,15264,137929,293993,110325,22093,55699,0
63,MARDIN,6,432253,682,106113,4820,13800,28510,4804,2076,242557,0
64,MERSIN,13,1206319,5404,295274,10721,143041,377350,144623,22013,160674,23998
65,MUGLA,7,708773,1953,169273,4259,42772,271295,122550,11947,23559,40431
66,MUS,3,183792,655,38269,5220,30195,0,10275,0,94704,0
67,NEVSEHIR,3,199220,1778,78757,4002,40585,34725,31921,2552,973,767
68,NIGDE,3,222259,1244,75314,5848,49847,53932,27442,3252,993,460
69,ORDU,6,500069,4054,227535,11459,63024,121914,49453,7093,992,1730
70,OSMANIYE,4,331910,1882,110641,6445,94925,57184,42311,5731,6624,940
71,RIZE,3,231469,3665,124491,11885,29939,49450,0,4050,689,683
72,SAKARYA,8,704094,6424,335205,35423,83999,114618,77725,18736,10268,3702
73,SAMSUN,9,891297,12224,380214,35770,126144,176283,108800,19545,2493,5583
74,SIIRT,3,160199,442,57159,1880,5804,11555,2272,0,76719,0
75,SINOP,2,144801,6477,62162,3255,12941,37495,13743,2166,494,1110
76,SIVAS,5,403906,41064,165234,13925,72782,65342,29554,7307,1723,1509
77,SANLIURFA,14,956187,4157,411638,28289,88632,72051,43645,0,242215,0
78,SIRNAK,4,264922,373,54728,1518,5682,21684,2837,3020,170695,0
79,TEKIRDAG,8,758560,4085,230112,11998,47306,279347,85690,17558,45995,11781
80,TOKAT,5,390636,11097,146619,9065,86501,80910,43657,5439,1197,1211
81,TRABZON,6,533167,8227,256125,24324,57206,95694,65444,9903,1597,1339
82,TUNCELI,1,56010,165,6815,547,3006,18067,1234,0,24571,0
83,USAK,3,247618,1619,88805,3494,22803,72810,43858,4025,3216,851
84,VAN,8,529624,4429,135592,16656,25514,43459,0,4101,287429,0
85,YALOVA,3,174301,1123,59356,3684,19620,50581,16781,4030,9789,1774
86,YOZGAT,4,255131,4199,112106,11216,58930,4,59685,3323,884,474
87,ZONGULDAK,5,392013,3338,156835,9850,32436,127274,39469,7064,1287,1171
"""

COL_MAP = {
    "AK PARTI": "AK PARTI", "CHP": "CHP", "MHP": "MHP",
    "IYI PARTI": "IYI PARTI", "DEM": "DEM PARTI", "DEM PARTI": "DEM PARTI",
    "YENIDEN REFAH": "YENIDEN REFAH", "ZAFER": "ZAFER", "ZAFER PARTISI": "ZAFER",
    "TIP": "TIP", "BBP": "BBP",
}

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_city_votes() -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(DATA_2023_CSV.strip()))
    df.columns = [c.strip() for c in df.columns]
    df["Milletvekili sayisi"] = pd.to_numeric(df["Milletvekili sayisi"], errors="coerce").fillna(0).astype(int)
    return df

def compute_national_totals_2023(df_votes: pd.DataFrame) -> Dict[str, float]:
    totals = {p: 0.0 for p in PARTIES_2023}
    for _, row in df_votes.iterrows():
        for csv_col, party_key in COL_MAP.items():
            if csv_col in df_votes.columns:
                val = pd.to_numeric(row[csv_col], errors="coerce")
                if not pd.isna(val):
                    totals[party_key] += float(val)
    return totals

# =============================================================================
# TRANSITION MATRIX MODEL + OPTIMIZATION
# =============================================================================
def get_initial_matrix() -> Dict[str, Dict[str, float]]:
    m = {}
    for p in PARTIES_2023:
        m[p] = {t: 0.0 for t in PARTIES}
        m[p][p] = 80.0
        m[p]["UNDECIDED"] = 10.0
        others = [t for t in PARTIES_2023 if t != p]
        share = 10.0 / max(1, len(others))
        for o in others:
            m[p][o] = share
    return m

def predict_national_counts(totals_2023: Dict[str, float], matrix: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    pred = {p: 0.0 for p in PARTIES}
    for src, src_votes in totals_2023.items():
        trans = matrix.get(src, {})
        for tgt, pct in trans.items():
            pred[tgt] += src_votes * (pct / 100.0)
    return pred

def rmse_to_targets(pred_counts: Dict[str, float], poll_targets: Dict[str, float], target_sum: float) -> float:
    valid_total = sum(pred_counts[p] for p in PARTIES_2023)
    if valid_total <= 0:
        return 1e9
    pred_pct = {p: (pred_counts[p] / valid_total) * target_sum for p in PARTIES_2023}
    errs = [(pred_pct[p] - poll_targets[p]) ** 2 for p in PARTIES_2023]
    return float(np.sqrt(np.mean(errs)))

def optimize_matrix(
    totals_2023: Dict[str, float],
    poll_targets: Dict[str, float],
    iterations: int = 12000
) -> Dict[str, Dict[str, float]]:
    target_sum = sum(poll_targets.values())
    best = get_initial_matrix()
    best_rmse = rmse_to_targets(predict_national_counts(totals_2023, best), poll_targets, target_sum)

    keys = list(best.keys())
    for _ in range(iterations):
        cand = copy.deepcopy(best)
        src = random.choice(keys)
        t1, t2 = random.sample(PARTIES, 2)
        delta = random.uniform(0.01, 1.25)

        if cand[src][t1] >= delta:
            cand[src][t1] -= delta
            cand[src][t2] += delta

            pred = predict_national_counts(totals_2023, cand)
            score = rmse_to_targets(pred, poll_targets, target_sum)

            if score < best_rmse:
                best_rmse = score
                best = cand

    return best

# =============================================================================
# EXACT CALIBRATION TO MATCH POLLS (TOTALS NOT FORCED TO 100)
# =============================================================================
def calibrate_party_factors(national_counts: Dict[str, float], poll_targets: Dict[str, float]) -> Dict[str, float]:
    target_sum = sum(poll_targets.values())
    valid_total = sum(national_counts[p] for p in PARTIES_2023)
    if valid_total <= 0:
        return {p: 1.0 for p in PARTIES_2023}

    current_pct = {p: (national_counts[p] / valid_total) * target_sum for p in PARTIES_2023}
    factors = {}
    for p in PARTIES_2023:
        factors[p] = (poll_targets[p] / current_pct[p]) if current_pct[p] > 0 else 1.0
    return factors

def apply_calibration_to_city_votes(city_vote_counts: Dict[str, float], factors: Dict[str, float]) -> Dict[str, float]:
    out = dict(city_vote_counts)
    for p in PARTIES_2023:
        out[p] = out.get(p, 0.0) * factors.get(p, 1.0)
    return out

# =============================================================================
# ALLIANCE + THRESHOLD LOGIC
# =============================================================================
def parties_exempt_from_threshold(alliances: List[Dict]) -> Set[str]:
    s: Set[str] = set()
    for a in alliances:
        for p in a["members"]:
            s.add(p)
    return s

def national_threshold_failures(
    national_counts: Dict[str, float],
    poll_targets: Dict[str, float],
    alliances: List[Dict],
) -> List[str]:
    target_sum = sum(poll_targets.values())
    valid_total = sum(national_counts[p] for p in PARTIES_2023)
    if valid_total <= 0:
        return []

    exempt = parties_exempt_from_threshold(alliances)
    failures = []
    for p in PARTIES_2023:
        pct = (national_counts[p] / valid_total) * target_sum
        if p not in exempt and pct < THRESHOLD_PERCENT:
            failures.append(p)
    return failures

# =============================================================================
# DHONDT (SAFE)
# =============================================================================
def dhondt(votes: Dict[str, float], seats: int) -> Dict[str, int]:
    if seats <= 0:
        return {k: 0 for k in votes}
    if not votes or max(votes.values(), default=0.0) <= 0:
        return {k: 0 for k in votes}

    seats_won = {k: 0 for k in votes}
    quot = {k: float(v) for k, v in votes.items()}

    for _ in range(seats):
        winner = max(quot, key=quot.get)
        if quot[winner] <= 0:
            break
        seats_won[winner] += 1
        quot[winner] = votes[winner] / (seats_won[winner] + 1)

    return seats_won

def allocate_mps_city(
    city_votes: Dict[str, float],
    seats: int,
    alliances: List[Dict],
    threshold_failed: List[str],
) -> Dict[str, int]:
    final = {p: 0 for p in PARTIES_2023}
    if seats <= 0:
        return final

    eligible = {p: float(city_votes.get(p, 0.0)) for p in PARTIES_2023}

    exempt = parties_exempt_from_threshold(alliances)
    for p in list(eligible.keys()):
        if p in threshold_failed and p not in exempt:
            eligible.pop(p, None)

    if not eligible or max(eligible.values(), default=0.0) <= 0:
        top_party = max(PARTIES_2023, key=lambda p: float(city_votes.get(p, 0.0)), default="AK PARTI")
        final[top_party] = seats
        return final

    stage1: Dict[str, float] = {}
    alliance_member_sets = {a["name"]: set(a["members"]) for a in alliances}

    used = set()
    for a in alliances:
        bloc_sum = sum(eligible.get(p, 0.0) for p in a["members"])
        if bloc_sum > 0:
            stage1[a["name"]] = bloc_sum
            used |= set(a["members"])

    for p, v in eligible.items():
        if p not in used and v > 0:
            stage1[p] = v

    if not stage1:
        top_party = max(eligible, key=eligible.get)
        final[top_party] = seats
        return final

    stage1_seats = dhondt(stage1, seats)
    allocated_stage1 = sum(stage1_seats.values())
    if allocated_stage1 < seats:
        top_bloc = max(stage1, key=stage1.get)
        stage1_seats[top_bloc] = stage1_seats.get(top_bloc, 0) + (seats - allocated_stage1)

    for k, s in stage1_seats.items():
        if s <= 0:
            continue

        if k in alliance_member_sets:
            members = alliance_member_sets[k]
            internal_votes = {p: eligible.get(p, 0.0) for p in members if eligible.get(p, 0.0) > 0}

            if not internal_votes:
                top_member = max(members, key=lambda p: eligible.get(p, 0.0) or float(city_votes.get(p, 0.0)), default=None)
                if top_member in final:
                    final[top_member] += s
                else:
                    final[max(PARTIES_2023, key=lambda p: float(city_votes.get(p, 0.0)), default="AK PARTI")] += s
                continue

            internal_seats = dhondt(internal_votes, s)
            allocated_internal = sum(internal_seats.values())
            if allocated_internal < s:
                top_member = max(internal_votes, key=internal_votes.get)
                internal_seats[top_member] += (s - allocated_internal)

            for p, ps in internal_seats.items():
                final[p] += ps
        else:
            if k in final:
                final[k] += s

    allocated_final = sum(final.values())
    if allocated_final < seats:
        top_party = max(eligible, key=eligible.get)
        final[top_party] += (seats - allocated_final)
    elif allocated_final > seats:
        top_party = max(final, key=final.get)
        final[top_party] -= (allocated_final - seats)

    return final

# =============================================================================
# EXCEL (IN-MEMORY)
# =============================================================================
def build_excel_bytes(vote_pct_by_city: pd.DataFrame, mp_by_city: pd.DataFrame, summary: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        vote_pct_by_city.to_excel(writer, sheet_name="vote_pct_by_city", index=False)
        mp_by_city.to_excel(writer, sheet_name="mp_by_city", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    return buf.getvalue()

# =============================================================================
# STREAMLIT UI
# =============================================================================
st.title("Turkish Election Simulator ")

df_votes = load_city_votes()
seats_by_city = dict(zip(df_votes["Il Adi"], df_votes["Milletvekili sayisi"]))
expected_total_seats = int(df_votes["Milletvekili sayisi"].sum())
totals_2023 = compute_national_totals_2023(df_votes)

# Default polls: start from 2023 national percentages scaled to 100
base_total = sum(totals_2023.values())
default_polls = {p: (totals_2023[p] / base_total) * 100.0 if base_total > 0 else 0.0 for p in PARTIES_2023}

with st.sidebar:
    st.header("Inputs")

    target_sum = st.number_input("Poll total scale (not forced to 100)", min_value=1.0, max_value=200.0, value=100.0, step=1.0)

    st.subheader("Polls (%)")
    polls: Dict[str, float] = {}
    for p in PARTIES_2023:
        polls[p] = st.number_input(p, value=float(default_polls[p]), step=0.1, format="%.2f")

    st.subheader("Alliances")
    st.caption("Parties in an alliance are exempt from threshold, and votes are aggregated for D'Hondt stage-1.")
    alliance_count = st.number_input("Number of alliances", min_value=0, max_value=10, value=0, step=1)

    alliances: List[Dict] = []
    for i in range(int(alliance_count)):
        name = st.text_input(f"Alliance {i+1} name", value=f"ALLIANCE_{i+1}", key=f"all_name_{i}")
        members = st.multiselect(f"Alliance {i+1} members", options=PARTIES_2023, default=[], key=f"all_mem_{i}")
        alliances.append({"name": name.strip().upper() or f"ALLIANCE_{i+1}", "members": [m.strip().upper() for m in members]})

    st.subheader("Model")
    iterations = st.slider("Optimization iterations", min_value=1000, max_value=50000, value=12000, step=1000)
    run_btn = st.button("Run simulation", type="primary")

if not run_btn:
    st.info("Set polls/alliances in the sidebar, then click **Run simulation**.")
    st.stop()

# Validate polls presence
for p in PARTIES_2023:
    if p not in polls:
        st.error(f"Missing poll entry: {p}")
        st.stop()

# Normalize poll targets to chosen target_sum (keeps relative proportions)
raw_sum = sum(polls.values())
if raw_sum <= 0:
    st.error("Poll total is <= 0. Enter positive poll values.")
    st.stop()

polls_scaled = {p: (polls[p] / raw_sum) * float(target_sum) for p in PARTIES_2023}

with st.spinner("Optimizing transition matrix..."):
    matrix = optimize_matrix(totals_2023, polls_scaled, iterations=int(iterations))

# Predict city counts
city_pred_counts = []
national_counts = {p: 0.0 for p in PARTIES}

for _, row in df_votes.iterrows():
    city = row["Il Adi"]

    raw_city_2023 = {p: 0.0 for p in PARTIES_2023}
    for csv_col, sim_key in COL_MAP.items():
        if csv_col in df_votes.columns:
            v = pd.to_numeric(row[csv_col], errors="coerce")
            if not pd.isna(v):
                raw_city_2023[sim_key] += float(v)

    pred_city = {p: 0.0 for p in PARTIES}
    for src, cnt in raw_city_2023.items():
        trans = matrix.get(src, {})
        for tgt, pct in trans.items():
            pred_city[tgt] += cnt * (pct / 100.0)

    for p in PARTIES:
        national_counts[p] += pred_city[p]

    city_pred_counts.append((city, pred_city))

# Calibrate to match polls exactly
factors = calibrate_party_factors(national_counts, polls_scaled)

calibrated_city_counts = []
calibrated_national_counts = {p: 0.0 for p in PARTIES}

for city, pred_city in city_pred_counts:
    cal = apply_calibration_to_city_votes(pred_city, factors)
    for p in PARTIES:
        calibrated_national_counts[p] += cal.get(p, 0.0)
    calibrated_city_counts.append((city, cal))

valid_total = sum(calibrated_national_counts[p] for p in PARTIES_2023)
if valid_total <= 0:
    st.error("Valid total is <= 0 after calibration; check inputs.")
    st.stop()

national_pct = {p: (calibrated_national_counts[p] / valid_total) * float(target_sum) for p in PARTIES_2023}
national_df = pd.DataFrame(
    [{"Party": p, "NationalPct": national_pct[p], "PollTarget": polls_scaled[p], "Diff": national_pct[p] - polls_scaled[p]}
     for p in PARTIES_2023]
).sort_values("PollTarget", ascending=False)

threshold_failed = national_threshold_failures(calibrated_national_counts, polls_scaled, alliances)

# Vote % by city
vote_rows = []
for city, cal in calibrated_city_counts:
    total_city_valid = sum(cal[p] for p in PARTIES_2023)
    row = {"City": city}
    for p in PARTIES_2023:
        row[p] = (cal[p] / total_city_valid) * 100.0 if total_city_valid > 0 else 0.0
    vote_rows.append(row)
vote_pct_by_city = pd.DataFrame(vote_rows)

# Allocate MPs by city
mp_rows = []
mp_totals = {p: 0 for p in PARTIES_2023}

for city, cal in calibrated_city_counts:
    seats = int(seats_by_city.get(city, 0))
    if seats <= 0:
        continue

    city_mps = allocate_mps_city(cal, seats, alliances, threshold_failed)

    city_sum = sum(city_mps.values())
    if city_sum != seats:
        top_party = max(PARTIES_2023, key=lambda p: float(cal.get(p, 0.0)), default="AK PARTI")
        city_mps[top_party] = city_mps.get(top_party, 0) + (seats - city_sum)

    row = {"City": city, "Seats": seats}
    for p in PARTIES_2023:
        row[p] = int(city_mps.get(p, 0))
        mp_totals[p] += int(city_mps.get(p, 0))
    mp_rows.append(row)

mp_by_city = pd.DataFrame(mp_rows)
mp_totals_series = pd.Series(mp_totals).sort_values(ascending=False)

# Alliance pct
alliance_vote = {}
for a in alliances:
    alliance_vote[a["name"]] = sum(national_pct.get(p, 0.0) for p in a["members"])
alliance_pct_series = pd.Series(alliance_vote).sort_values(ascending=False) if alliance_vote else pd.Series(dtype=float)

# =============================================================================
# DISPLAY
# =============================================================================
col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.subheader("National results (constructed to match polls)")
    st.dataframe(national_df, use_container_width=True)

    if threshold_failed:
        st.warning(f"Threshold failed (excluded unless alliance member): {', '.join(sorted(threshold_failed))}")
    else:
        st.success("No parties failed the threshold (outside alliances).")

with col2:
    st.subheader("Seat check")
    got_total = int(mp_totals_series.sum())
    st.metric("Allocated seats", got_total)
    st.metric("Expected seats (from data)", expected_total_seats)
    if got_total != expected_total_seats:
        st.error("Seat totals do NOT match the seat map. Check alliances / inputs.")
    else:
        st.success("Seat totals match the seat map.")

st.subheader("Projected parliament seats")
st.dataframe(mp_totals_series.rename("Seats").reset_index().rename(columns={"index": "Party"}), use_container_width=True)

# Charts (inline)
st.subheader("Charts")
c1, c2, c3 = st.columns(3)

with c1:
    fig, ax = plt.subplots()
    pd.Series(national_pct).sort_values(ascending=False).plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("National Vote Percent")
    ax.set_ylabel("Percent")

    # Increase Y margin so text fits
    ax.margins(y=0.2)

    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", padding=3)

    plt.tight_layout()
    st.pyplot(fig)

with c2:
    fig, ax = plt.subplots()
    mp_totals_series.sort_values(ascending=False).plot(kind="bar", ax=ax, color="salmon")
    ax.set_title("Projected MP Seats by Party")
    ax.set_ylabel("Seats")

    # Increase Y margin
    ax.margins(y=0.2)

    for c in ax.containers:
        ax.bar_label(c, fmt="%d", padding=3)

    plt.tight_layout()
    st.pyplot(fig)

with c3:
    if len(alliance_pct_series) > 0:
        fig, ax = plt.subplots()
        alliance_pct_series.plot(kind="bar", ax=ax, color="lightgreen")
        ax.set_title("Alliance Vote Percent")
        ax.set_ylabel("Percent")

        # Increase Y margin
        ax.margins(y=0.2)

        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", padding=3)

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No alliances defined (no alliance chart).")

# Excel download
party_to_alliance = {}
for a in alliances:
    for m in a["members"]:
        party_to_alliance[m] = a["name"]

summary_df = pd.DataFrame(
    [{
        "Party": p,
        "NationalPct": float(national_pct[p]),
        "PollTarget": float(polls_scaled[p]),
        "Diff": float(national_pct[p] - polls_scaled[p]),
        "MPSeats": int(mp_totals[p]),
        "Alliance": party_to_alliance.get(p, ""),
    } for p in PARTIES_2023]
)

xlsx_bytes = build_excel_bytes(vote_pct_by_city, mp_by_city, summary_df)

st.download_button(
    label="Download simulation_outputs.xlsx",
    data=xlsx_bytes,
    file_name="simulation_outputs.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# =============================================================================
# NEW: DISPLAY TRANSITION MATRIX
# =============================================================================
st.divider()
st.subheader("🔍 Optimization Result: Transition Matrix")
st.markdown("""
**How to read:** - **Rows** are the source parties (2023).
- **Columns** are where those votes went (Target).
- *Example: A value of 5.0 in row 'AKP' and column 'CHP' means 5% of 2023 AKP voters switched to CHP.*
""")

# Convert the dictionary to a DataFrame for display
matrix_df = pd.DataFrame(matrix).T.fillna(0)
# Reorder columns to match standard party list
matrix_df = matrix_df[PARTIES]

# Display as an interactive table
st.dataframe(matrix_df.style.format("{:.1f}%").background_gradient(cmap="Reds", axis=1), use_container_width=True)

# OPTIONAL: Heatmap Visualization
fig_matrix, ax_matrix = plt.subplots(figsize=(10, 8))
# We use a simple matplotlib imshow since we don't have seaborn installed
cax = ax_matrix.imshow(matrix_df, cmap='viridis', aspect='auto')
ax_matrix.set_xticks(range(len(matrix_df.columns)))
ax_matrix.set_xticklabels(matrix_df.columns, rotation=90)
ax_matrix.set_yticks(range(len(matrix_df.index)))
ax_matrix.set_yticklabels(matrix_df.index)
ax_matrix.set_title("Voter Flow Heatmap (Source -> Target)")
fig_matrix.colorbar(cax)

# Add text annotations
for i in range(len(matrix_df.index)):
    for j in range(len(matrix_df.columns)):
        val = matrix_df.iloc[i, j]
        if val > 1.0: # Only show numbers > 1% to keep it clean
            text = ax_matrix.text(j, i, f"{val:.0f}",
                           ha="center", va="center", color="w", fontsize=8)

st.pyplot(fig_matrix)
st.divider()
