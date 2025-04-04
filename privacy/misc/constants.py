import pandas as pd
from pandas import Timestamp

mapping_code_hospital_short_name_dict = [
    ("014", "APR"),
    ("028", "ABC"),
    ("095", "AVC"),
    ("005", "BJN"),
    ("009", "BRK"),
    ("010", "BCT"),
    ("011", "BCH"),
    ("033", "BRT"),
    ("016", "BRC"),
    ("042", "CFX"),
    ("021", "CCH"),
    ("022", "CCL"),
    ("029", "ERX"),
    ("036", "GCL"),
    ("075", "EGC"),
    ("038", "HND"),
    ("026", "HMN"),
    ("099", "HAD"),
    ("041", "HTD"),
    ("032", "JVR"),
    ("044", "JFR"),
    ("047", "LRB"),
    ("049", "LRG"),
    ("053", "LMR"),
    ("061", "NCK"),
    ("096", "PBR"),
    ("066", "PSL"),
    ("068", "RPC"),
    ("069", "RMB"),
    ("070", "RDB"),
    ("072", "RTH"),
    ("073", "SAT"),
    ("079", "SPR"),
    ("076", "SLS"),
    ("084", "SSL"),
    ("087", "TNN"),
    ("088", "TRS"),
    ("090", "VGR"),
    ("064", "VPD"),
    ("INC", "INC"),
]

mapping_code_hospital_short_name = pd.DataFrame.from_records(
    mapping_code_hospital_short_name_dict,
    columns=["care_site_source_value", "care_site_short_name"],
)


d_v = {
    "person_id": {
        0: 1,
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 2,
        6: 3,
        7: 3,
        8: 3,
        9: 4,
        10: 4,
        11: 4,
        12: 5,
        13: 6,
        14: 7,
    },
    "visit_start_date": {
        0: Timestamp("2019-05-01 00:00:00"),
        1: Timestamp("2019-05-15 00:00:00"),
        2: Timestamp("2019-07-01 00:00:00"),
        3: Timestamp("2019-05-02 00:00:00"),
        4: Timestamp("2019-05-14 00:00:00"),
        5: Timestamp("2019-07-04 00:00:00"),
        6: Timestamp("2019-05-02 00:00:00"),
        7: Timestamp("2019-05-16 00:00:00"),
        8: Timestamp("2019-07-02 00:00:00"),
        9: Timestamp("2021-05-02 00:00:00"),
        10: Timestamp("2021-05-16 00:00:00"),
        11: Timestamp("2021-07-02 00:00:00"),
        12: Timestamp("2021-07-02 00:00:00"),
        13: Timestamp("2021-07-15 00:00:00"),
        14: Timestamp("2021-07-15 00:00:00"),
    },
    "visit_end_date": {
        0: Timestamp("2019-05-02 00:00:00"),
        1: Timestamp("2019-05-17 00:00:00"),
        2: Timestamp("2019-07-04 00:00:00"),
        3: Timestamp("2019-05-03 00:00:00"),
        4: Timestamp("2019-05-16 00:00:00"),
        5: Timestamp("2019-07-07 00:00:00"),
        6: Timestamp("2019-05-03 00:00:00"),
        7: Timestamp("2019-05-18 00:00:00"),
        8: Timestamp("2019-07-05 00:00:00"),
        9: Timestamp("2021-05-03 00:00:00"),
        10: Timestamp("2021-05-18 00:00:00"),
        11: Timestamp("2021-07-05 00:00:00"),
        12: Timestamp("2021-07-05 00:00:00"),
        13: Timestamp("2021-07-18 00:00:00"),
        14: Timestamp("2021-07-18 00:00:00"),
    },
    "care_site_short_name": {
        0: "B",
        1: "A",
        2: "B",
        3: "B",
        4: "B",
        5: "A",
        6: "B",
        7: "A",
        8: "B",
        9: "B",
        10: "A",
        11: "B",
        12: "B",
        13: "B",
        14: "B",
    },
}

d_c = {
    "person_id": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7},
    "birth_date": {
        0: Timestamp("2010-05-02 00:00:00"),
        1: Timestamp("2010-05-15 00:00:00"),
        2: Timestamp("2010-05-02 00:00:00"),
        3: Timestamp("2010-05-02 00:00:00"),
        4: Timestamp("2010-05-02 00:00:00"),
        5: Timestamp("2010-05-02 00:00:00"),
        6: Timestamp("2010-05-02 00:00:00"),
    },
    "gender_source_value": {
        0: "m",
        1: "m",
        2: "m",
        3: "m",
        4: "f",
        5: "f",
        6: "f",
    },
    "death_date": {
        0: Timestamp(None),
        1: Timestamp(None),
        2: Timestamp(None),
        3: Timestamp(None),
        4: Timestamp("2018-05-02 00:00:00"),
        5: Timestamp("2018-05-02 00:00:00"),
        6: Timestamp("2018-05-15 00:00:00"),
    },
}

toy_stays = pd.DataFrame(d_v)
toy_cohort = pd.DataFrame(d_c)

cohort_name_mapping = {
    "all_population": "Overall",
    "bronchiolitis": "Seasonal bronchiolitis",
    "seasonal_flu": "Seasonal flu",
    "bariatric_surgery": "Bariatric surgery readmission",
    "cancer": "Cancer",
    "pancreatic_cancer": "Pancreatic Cancer",
}

colors_cohorts = {
    "bronchiolitis": "blue",
    "seasonal_flu": "red",
    "bariatric_surgery": "green",
    "cancer": "#FFC300",
    "pancreatic_cancer": "purple",
    "all_population": "black",
}

inversed_mapping = {value: key for key, value in cohort_name_mapping.items()}

colors_cohorts_inverse_mapping = {
    key: colors_cohorts[value] for key, value in inversed_mapping.items()
}

attack_knowledge_name_mapping = {
    "visit_start_date + visit_end_date + birth_date + death_date + gender + hospital": "Sex, Date of Birth,\nDate of Death, Admission Dates,\nDischarge Dates, Hospital",
    "visit_start_date + visit_end_date + birth_date + death_date + gender": "Sex, Date of Birth,\nDate of Death, Admission Dates,\nDischarge Dates",
    "visit_start_date + birth_date + death_date + gender": "Sex, Date of Birth,\nDate of Death, Admission Dates",
    "birth_date + death_date + gender": "Sex, Date of Birth, Date of Death",
}

pseudonimizer_name_mapping = {
    "NoPseudonymizer": "No pseudonymisation",
    "BasePseudonymizer": "Base pseudonymisation",
    "BirthPseudonymizer": "Birth pseudonymisation",
    "StayPseudonymizer": "Hospital stay pseudonymisation",
}
