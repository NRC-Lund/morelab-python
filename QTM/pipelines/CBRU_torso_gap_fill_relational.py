from .gap_fill_relational import gap_fill_relational


base_marker_names = [
    "LAC",
    "RAC",
    "SN",
    "XP",
    "C7",
    "RSCAP",
    "LSCAP",
    "LBUR",
    "LBLR",
    "LBLL",
    "LBUL"]

base_gap_fill_rules = {

    "RAC": [
        # Strongest: opposite shoulder + central upper torso
        ["LAC", "C7"],
        ["LAC", "SN"],

        # Shoulder + central torso
        ["LAC", "XP"],
        ["C7", "RSCAP"],
        ["SN", "RSCAP"],

        # Same-side scapula + opposite shoulder
        ["RSCAP", "LAC"],

        # Same-side / opposite-side scapular geometry
        ["RSCAP", "LSCAP"],
    ],



    "LAC": [
        # Strongest: opposite shoulder + central upper torso
        ["RAC", "C7"],
        ["RAC", "SN"],

        # Shoulder + central torso
        ["RAC", "XP"],
        ["C7", "LSCAP"],
        ["SN", "LSCAP"],

        # Same-side scapula + opposite shoulder
        ["LSCAP", "RAC"],

        # Bilateral scapular geometry
        ["RSCAP", "LSCAP"],
    ],


    "SN": [
        # Very strong vertical torso axis
        ["C7", "XP"],

        # Shoulder-width information
        ["LAC", "RAC"],

        # Acromion + C7
        ["RAC", "C7"],
        ["LAC", "C7"],

        # Acromion + xiphoid
        ["RAC", "XP"],
        ["LAC", "XP"],

        # Scapula + opposite shoulder
        ["RSCAP", "LAC"],
        ["LSCAP", "RAC"],

        # Bilateral scapular geometry
        ["RSCAP", "LSCAP"],
    ],


    "XP": [
        # Strongest: sternum axis
        ["SN", "C7"],

        # Shoulder-width / torso geometry
        ["RAC", "LAC"],
        ["RAC", "SN"],
        ["LAC", "SN"],

        # Acromion + C7
        ["RAC", "C7"],
        ["LAC", "C7"],

        # Scapular support
        ["RSCAP", "LSCAP"],
        ["RSCAP", "C7"],
        ["LSCAP", "C7"],
    ],


    "C7": [
        # Strongest central vertical axis
        ["SN", "XP"],

        # Shoulder landmarks
        ["LAC", "RAC"],
        ["RAC", "SN"],
        ["LAC", "SN"],
        ["RAC", "XP"],
        ["LAC", "XP"],

        # Scapular landmarks
        ["RSCAP", "LSCAP"],
        ["RSCAP", "SN"],
        ["LSCAP", "SN"],
        ["RSCAP", "XP"],
        ["LSCAP", "XP"],
    ],


    "RSCAP": [
        # Best bilateral shoulder/scapular geometry
        ["LSCAP", "RAC"],
        ["LSCAP", "LAC"],
        ["LSCAP", "C7"],

        # Same-side shoulder + central torso
        ["RAC", "C7"],
        ["RAC", "SN"],
        ["RAC", "XP"],

        # Opposite shoulder + central torso
        ["LAC", "C7"],
        ["LAC", "SN"],

        # Bilateral scapulae
        ["LSCAP", "C7"],
    ],


    "LSCAP": [
        # Best bilateral shoulder/scapular geometry
        ["RSCAP", "LAC"],
        ["RSCAP", "RAC"],
        ["RSCAP", "C7"],

        # Same-side shoulder + central torso
        ["LAC", "C7"],
        ["LAC", "SN"],
        ["LAC", "XP"],

        # Opposite shoulder + central torso
        ["RAC", "C7"],
        ["RAC", "SN"],

        # Bilateral scapulae
        ["RSCAP", "C7"],
    ],


    "LBUR": [
        # Central spine + lower back
        ["C7", "LBLR"],
        ["C7", "LBLL"],
        ["C7", "LBUL"],

        # Bilateral lower-back geometry
        ["LBLR", "LBLL"],
        ["LBLR", "LBUL"],
        ["LBLL", "LBUL"],

        # Upper torso support
        ["RSCAP", "LBLR"],
        ["LSCAP", "LBLL"],
        ["RSCAP", "C7"],
        ["LSCAP", "C7"],
    ],


    "LBLR": [
        # Same-region back geometry
        ["LBUR", "LBLL"],
        ["LBUR", "LBUL"],
        ["LBLL", "LBUL"],

        # Spine-to-lower-back
        ["C7", "LBUR"],
        ["C7", "LBUL"],

        # Scapular support
        ["RSCAP", "LBUR"],
        ["LSCAP", "LBUL"],
    ],


    "LBLL": [
        # Same-region back geometry
        ["LBUL", "LBUR"],
        ["LBUL", "LBLR"],
        ["LBUR", "LBLR"],

        # Spine-to-lower-back
        ["C7", "LBUR"],
        ["C7", "LBUL"],

        # Scapular support
        ["LSCAP", "LBUR"],
        ["RSCAP", "LBUL"],
    ],


    "LBUL": [
        # Same-region back geometry
        ["LBUR", "LBLL"],
        ["LBUR", "LBLR"],
        ["LBLL", "LBLR"],

        # Spine-to-lower-back
        ["C7", "LBUR"],
        ["C7", "LBLL"],

        # Scapular support
        ["RSCAP", "LBUR"],
        ["LSCAP", "LBLL"],
    ],
}

def CBRU_torso_gap_fill_relational():
    gap_fill_relational(base_marker_names, base_gap_fill_rules)
