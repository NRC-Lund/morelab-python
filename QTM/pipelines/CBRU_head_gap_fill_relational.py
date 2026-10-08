from .gap_fill_relational import gap_fill_relational


base_marker_names = [
    "RFHD",
    "RBHD",
    "LBHD",
    "LFHD"]

base_gap_fill_rules = {

    "RFHD": [
        # Same-side front/back + opposite-side front
        ["RBHD", "LFHD"],

        # Same-side front/back + opposite-side back
        ["RBHD", "LBHD"],

        # Opposite-side front/back
        ["LFHD", "LBHD"],

        # Three-point configuration
        ["RBHD", "LFHD", "LBHD"],
    ],

    "RBHD": [
        # Same-side front/back + opposite-side back
        ["RFHD", "LBHD"],

        # Same-side front/back + opposite-side front
        ["RFHD", "LFHD"],

        # Opposite-side front/back
        ["LFHD", "LBHD"],

        # Three-point configuration
        ["RFHD", "LFHD", "LBHD"],
    ],

    "LBHD": [
        # Same-side front/back + opposite-side back
        ["LFHD", "RBHD"],

        # Same-side front/back + opposite-side front
        ["LFHD", "RFHD"],

        # Opposite-side front/back
        ["RFHD", "RBHD"],

        # Three-point configuration
        ["LFHD", "RFHD", "RBHD"],
    ],

    "LFHD": [
        # Same-side front/back + opposite-side front
        ["LBHD", "RFHD"],

        # Same-side front/back + opposite-side back
        ["LBHD", "RBHD"],

        # Opposite-side front/back
        ["RFHD", "RBHD"],

        # Three-point configuration
        ["LBHD", "RFHD", "RBHD"],
    ],
}

def CBRU_head_gap_fill_relational():
    gap_fill_relational(base_marker_names, base_gap_fill_rules)
