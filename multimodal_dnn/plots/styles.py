model_styles = {
    'Blosum_crosstrain': {
        'group': 0,
        'color': '#BDA048',
        'label': 'BLOSUM62 (crosstrain)'
    },
    'Blosum_final': {
        'group': 0,
        'color': '#A37B3C',
        'label': 'BLOSUM62 (test)'
    },
    'AAindex_crosstrain': {
        'group': 1,
        'color': '#1D981D',
        'label': 'AAindex (crosstrain)'
    },
    'AAindex_final': {
        'group': 1,
        'color': '#066B06',
        'label': 'AAindex (test)'
    },
    'Combined_crosstrain': {
        'group': 2,
        'color': '#2573AA',
        'label': 'Combined (crosstrain)'
    },
    'Combined_final': {
        'group': 2,
        'color': '#0E4F7D',
        'label': 'Combined (test)'
    },
}

class_styles = {
    'NO_SP': {
        'label': 'No SP',
    },
    'SP': {
        'label': 'Sec/SPI',
    },
    'LIPO': {
        'label': 'Sec/SPII',
    },
    'TAT': {
        'label': 'Tat/SPI',
    },
    'INNER': {
        'label': 'Intracellular',
    },
    'IN': {
        'label': 'Intracellular',
    },
    'MEMBR': {
        'label': 'Transmembrane',
    },
    'TM_IN_OUT': {
        'label': 'Transmembr. (in to out)',
    },
    'TM_OUT_IN': {
        'label': 'Transmembr. (out to in)',
    },
    'OUTER': {
        'label': 'Extracellular',
    },
    'OUT': {
        'label': 'Extracellular',
    },
    'SP_CLEAVE': {
        'label': 'Sec/SPI Cleav.S.',
    },
    'LIPO_CLEAVE': {
        'label': 'Sec/SPII Cleav.S.',
    },
}

logo_styles = {
    'Blosum_final': {
        'xlims': {
            0: [-1, 70],
            1: [-35, 5],
            2: [-25, 5],
            3: [-50, 5],
        },
        'xlabels': {
            0: 'Position',
            1: 'Position relative to cleavage site',
            2: 'Position relative to cleavage site',
            3: 'Position relative to cleavage site',
        }
    },
    'Combined_final': {
        'xlims': {
            0: [-1, 70],
            1: [-35, 55],
            2: [-20, 20],
            3: [-60, 40],
        },
        'xlabels': {
            0: 'Position',
            1: 'Position relative to cleavage site',
            2: 'Position relative to cleavage site',
            3: 'Position relative to cleavage site',
        }
    }
}

aaindex_styles = {
    'plot_features': [
        'KLEP840101', # Net charge (Klein et al., 1984)
        'RADA880108', # Mean polarity (Radzicka-Wolfenden, 1988)
        'KYTJ820101', # Hydropathy index (Kyte-Doolittle, 1982)
        # Normalized van der Waals volume (Fauchere et al., 1988)
        'FAUJ880103',
    ],
    'AAindex_final': {
        'aaindex_ids': sorted([
            # Normalized frequency of alpha-helix (Chou-Fasman, 1978b)
            'CHOP780201',
            # Normalized frequency of beta-sheet (Chou-Fasman, 1978b)
            'CHOP780202',
            # Normalized frequency of beta-turn (Chou-Fasman, 1978b)
            'CHOP780203',
            # Normalized van der Waals volume (Fauchere et al., 1988)
            'FAUJ880103',
            'KLEP840101', # Net charge (Klein et al., 1984)
            'KYTJ820101', # Hydropathy index (Kyte-Doolittle, 1982)
            'MITS020101', # Amphiphilicity index (Mitaku et al., 2002)
            'RADA880108', # Mean polarity (Radzicka-Wolfenden, 1988)
            'CHAM810101', # Steric parameter (Charton, 1981)
            'CHAM830107', # A parameter of charge transfer capability (Charton-Charton, 1983)
            'JANJ780101', # Average accessible surface area (Janin et al., 1978)
            'MEIH800103', # Average side chain orientation angle (Meirovitch et al., 1980)
            'VELV850101', # Electron-ion interaction potential (Veljkovic et al., 1985)
            'WERD780101', # Propensity to be buried inside (Wertz-Scheraga, 1978)
            'ZIMJ680105', # RF rank (Zimmerman et al., 1968)
            'ZIMJ680104', # Isoelectric point (Zimmerman et al., 1968)
        ]),
        'xlims': {
            0: [-1, 70],
            1: [-35, 50],
            2: [-25, 25],
            3: [-50, 25],
        },
        'xlabels': {
            0: 'Position',
            1: 'Position relative to cleavage site',
            2: 'Position relative to cleavage site',
            3: 'Position relative to cleavage site',
        },
        'ylabels': {
            'CHOP780201': 'Normalized frequency of alpha-helix (Chou-Fasman, 1978b)',
            'CHOP780202': 'Normalized frequency of beta-sheet (Chou-Fasman, 1978b)',
            'CHOP780203': 'Normalized frequency of beta-turn (Chou-Fasman, 1978b)',
            'FAUJ880103': 'Normalized van der Waals volume (Fauchere et al., 1988)',
            'KLEP840101': 'Net charge (Klein et al., 1984)',
            'KYTJ820101': 'Hydropathy index (Kyte-Doolittle, 1982)',
            'MITS020101': 'Amphiphilicity index (Mitaku et al., 2002)',
            'RADA880108': 'Mean polarity (Radzicka-Wolfenden, 1988)',
            'CHAM810101': 'Steric parameter (Charton, 1981)',
            'CHAM830107': 'A parameter of charge transfer capability (Charton-Charton, 1983)',
            'JANJ780101': 'Average accessible surface area (Janin et al., 1978)',
            'MEIH800103': 'Average side chain orientation angle (Meirovitch et al., 1980)',
            'VELV850101': 'Electron-ion interaction potential (Veljkovic et al., 1985)',
            'WERD780101': 'Propensity to be buried inside (Wertz-Scheraga, 1978)',
            'ZIMJ680105': 'RF rank (Zimmerman et al., 1968)',
            'ZIMJ680104': 'Isoelectric point (Zimmerman et al., 1968)',
        }
    },
    'Combined_final': {
        'aaindex_ids': sorted([
            # Normalized frequency of alpha-helix (Chou-Fasman, 1978b)
            'CHOP780201',
            # Normalized frequency of beta-sheet (Chou-Fasman, 1978b)
            'CHOP780202',
            # Normalized frequency of beta-turn (Chou-Fasman, 1978b)
            'CHOP780203',
            # Normalized van der Waals volume (Fauchere et al., 1988)
            'FAUJ880103',
            'KLEP840101', # Net charge (Klein et al., 1984)
            'KYTJ820101', # Hydropathy index (Kyte-Doolittle, 1982)
            'MITS020101', # Amphiphilicity index (Mitaku et al., 2002)
            'RADA880108', # Mean polarity (Radzicka-Wolfenden, 1988)
            'CHAM810101', # Steric parameter (Charton, 1981)
            'CHAM830107', # A parameter of charge transfer capability (Charton-Charton, 1983)
            'JANJ780101', # Average accessible surface area (Janin et al., 1978)
            'MEIH800103', # Average side chain orientation angle (Meirovitch et al., 1980)
            'VELV850101', # Electron-ion interaction potential (Veljkovic et al., 1985)
            'WERD780101', # Propensity to be buried inside (Wertz-Scheraga, 1978)
            'ZIMJ680105', # RF rank (Zimmerman et al., 1968)
            'ZIMJ680104', # Isoelectric point (Zimmerman et al., 1968)
        ]),
        'ylabels': {
            'CHOP780201': 'Normalized frequency of alpha-helix (Chou-Fasman, 1978b)',
            'CHOP780202': 'Normalized frequency of beta-sheet (Chou-Fasman, 1978b)',
            'CHOP780203': 'Normalized frequency of beta-turn (Chou-Fasman, 1978b)',
            'FAUJ880103': 'Normalized van der Waals volume (Fauchere et al., 1988)',
            'KLEP840101': 'Net charge (Klein et al., 1984)',
            'KYTJ820101': 'Hydropathy index (Kyte-Doolittle, 1982)',
            'MITS020101': 'Amphiphilicity index (Mitaku et al., 2002)',
            'RADA880108': 'Mean polarity (Radzicka-Wolfenden, 1988)',
            'CHAM810101': 'Steric parameter (Charton, 1981)',
            'CHAM830107': 'A parameter of charge transfer capability (Charton-Charton, 1983)',
            'JANJ780101': 'Average accessible surface area (Janin et al., 1978)',
            'MEIH800103': 'Average side chain orientation angle (Meirovitch et al., 1980)',
            'VELV850101': 'Electron-ion interaction potential (Veljkovic et al., 1985)',
            'WERD780101': 'Propensity to be buried inside (Wertz-Scheraga, 1978)',
            'ZIMJ680105': 'RF rank (Zimmerman et al., 1968)',
            'ZIMJ680104': 'Isoelectric point (Zimmerman et al., 1968)',
        },
        'xlims': {
            0: [-1, 70],
            1: [-35, 55],
            2: [-20, 20],
            3: [-60, 40],
        },
        'xlabels': {
            0: 'Position',
            1: 'Position relative to cleavage site',
            2: 'Position relative to cleavage site',
            3: 'Position relative to cleavage site',
        }
    }
}

kingdom_styles = {
    'EUKARYA': {
        'label': 'Eukarya',
    },
    'ARCHAEA': {
        'label': 'Archaea',
    },
    'POSITIVE': {
        'label': 'Gram positive',
    },
    'NEGATIVE': {
        'label': 'Gram negative',
    },
}