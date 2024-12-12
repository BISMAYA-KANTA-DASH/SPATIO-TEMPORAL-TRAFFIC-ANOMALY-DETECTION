import json
import pandas as pd

# Load JSON data
data = '''
{
  "0RJPQ_97dcs_000387": {
    "video_start": 387,
    "video_end": 506,
    "anomaly_start": 41,
    "anomaly_end": 94,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 120,
    "subset": "val"
  },
  "0RJPQ_97dcs_002109": {
    "video_start": 2109,
    "video_end": 2192,
    "anomaly_start": 21,
    "anomaly_end": 44,
    "anomaly_class": "ego: turning",
    "num_frames": 84,
    "subset": "val"
  },
  "0RJPQ_97dcs_002604": {
    "video_start": 2604,
    "video_end": 2737,
    "anomaly_start": 12,
    "anomaly_end": 92,
    "anomaly_class": "ego: lateral",
    "num_frames": 134,
    "subset": "val"
  },
  "0RJPQ_97dcs_002834": {
    "video_start": 2834,
    "video_end": 2959,
    "anomaly_start": 59,
    "anomaly_end": 126,
    "anomaly_class": "ego: unknown",
    "num_frames": 126,
    "subset": "val"
  },
  "0RJPQ_97dcs_003475": {
    "video_start": 3475,
    "video_end": 3601,
    "anomaly_start": 42,
    "anomaly_end": 126,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 127,
    "subset": "val"
  },
  "0RJPQ_97dcs_003861": {
    "video_start": 3861,
    "video_end": 3931,
    "anomaly_start": 47,
    "anomaly_end": 65,
    "anomaly_class": "ego: turning",
    "num_frames": 71,
    "subset": "val"
  },
  "0RJPQ_97dcs_004443": {
    "video_start": 4443,
    "video_end": 4501,
    "anomaly_start": 10,
    "anomaly_end": 43,
    "anomaly_class": "other: lateral",
    "num_frames": 59,
    "subset": "val"
  },
  "0RJPQ_97dcs_004503": {
    "video_start": 4503,
    "video_end": 4563,
    "anomaly_start": 26,
    "anomaly_end": 53,
    "anomaly_class": "ego: turning",
    "num_frames": 61,
    "subset": "val"
  },
  "0RJPQ_97dcs_004565": {
    "video_start": 4565,
    "video_end": 4699,
    "anomaly_start": 22,
    "anomaly_end": 127,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 135,
    "subset": "val"
  },
  "0RJPQ_97dcs_004791": {
    "video_start": 4791,
    "video_end": 4865,
    "anomaly_start": 23,
    "anomaly_end": 44,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 75,
    "subset": "val"
  },
  "0RJPQ_97dcs_004957": {
    "video_start": 4957,
    "video_end": 5038,
    "anomaly_start": 37,
    "anomaly_end": 62,
    "anomaly_class": "ego: lateral",
    "num_frames": 82,
    "subset": "val"
  },
  "0RJPQ_97dcs_005223": {
    "video_start": 5223,
    "video_end": 5307,
    "anomaly_start": 49,
    "anomaly_end": 76,
    "anomaly_class": "ego: turning",
    "num_frames": 85,
    "subset": "val"
  },
  "0qfbmt4G8Rw_001201": {
    "video_start": 1201,
    "video_end": 1333,
    "anomaly_start": 63,
    "anomaly_end": 121,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 133,
    "subset": "val"
  },
  "0qfbmt4G8Rw_001335": {
    "video_start": 1335,
    "video_end": 1415,
    "anomaly_start": 30,
    "anomaly_end": 59,
    "anomaly_class": "other: turning",
    "num_frames": 81,
    "subset": "val"
  },
  "0qfbmt4G8Rw_001602": {
    "video_start": 1602,
    "video_end": 1751,
    "anomaly_start": 75,
    "anomaly_end": 111,
    "anomaly_class": "other: turning",
    "num_frames": 150,
    "subset": "val"
  },
  "0qfbmt4G8Rw_002476": {
    "video_start": 2476,
    "video_end": 2531,
    "anomaly_start": 50,
    "anomaly_end": 56,
    "anomaly_class": "ego: turning",
    "num_frames": 56,
    "subset": "val"
  },
  "0qfbmt4G8Rw_004485": {
    "video_start": 4485,
    "video_end": 4573,
    "anomaly_start": 37,
    "anomaly_end": 55,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "0qfbmt4G8Rw_004658": {
    "video_start": 4658,
    "video_end": 4756,
    "anomaly_start": 34,
    "anomaly_end": 93,
    "anomaly_class": "other: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "0z7J7rOpTic_000441": {
    "video_start": 441,
    "video_end": 550,
    "anomaly_start": 25,
    "anomaly_end": 49,
    "anomaly_class": "ego: turning",
    "num_frames": 110,
    "subset": "val"
  },
  "0z7J7rOpTic_000662": {
    "video_start": 662,
    "video_end": 750,
    "anomaly_start": 29,
    "anomaly_end": 48,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 89,
    "subset": "val"
  },
  "0z7J7rOpTic_000857": {
    "video_start": 857,
    "video_end": 951,
    "anomaly_start": 40,
    "anomaly_end": 60,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 95,
    "subset": "val"
  },
  "0z7J7rOpTic_000953": {
    "video_start": 953,
    "video_end": 1071,
    "anomaly_start": 27,
    "anomaly_end": 54,
    "anomaly_class": "other: turning",
    "num_frames": 119,
    "subset": "val"
  },
  "0z7J7rOpTic_001386": {
    "video_start": 1386,
    "video_end": 1494,
    "anomaly_start": 24,
    "anomaly_end": 42,
    "anomaly_class": "ego: lateral",
    "num_frames": 109,
    "subset": "val"
  },
  "0z7J7rOpTic_001971": {
    "video_start": 1971,
    "video_end": 2064,
    "anomaly_start": 60,
    "anomaly_end": 94,
    "anomaly_class": "ego: start_stop_or_stationary",
    "num_frames": 94,
    "subset": "val"
  },
  "0z7J7rOpTic_002341": {
    "video_start": 2341,
    "video_end": 2489,
    "anomaly_start": 37,
    "anomaly_end": 77,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 149,
    "subset": "val"
  },
  "0z7J7rOpTic_002491": {
    "video_start": 2491,
    "video_end": 2591,
    "anomaly_start": 24,
    "anomaly_end": 43,
    "anomaly_class": "other: turning",
    "num_frames": 101,
    "subset": "val"
  },
  "0z7J7rOpTic_006011": {
    "video_start": 6011,
    "video_end": 6129,
    "anomaly_start": 24,
    "anomaly_end": 38,
    "anomaly_class": "other: pedestrian",
    "num_frames": 119,
    "subset": "val"
  },
  "1u69z-wsDIc_000565": {
    "video_start": 565,
    "video_end": 673,
    "anomaly_start": 34,
    "anomaly_end": 69,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 109,
    "subset": "val"
  },
  "1u69z-wsDIc_000785": {
    "video_start": 785,
    "video_end": 911,
    "anomaly_start": 35,
    "anomaly_end": 102,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 127,
    "subset": "val"
  },
  "1u69z-wsDIc_000913": {
    "video_start": 913,
    "video_end": 1020,
    "anomaly_start": 44,
    "anomaly_end": 91,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 108,
    "subset": "val"
  },
  "1u69z-wsDIc_002679": {
    "video_start": 2679,
    "video_end": 2767,
    "anomaly_start": 39,
    "anomaly_end": 71,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "1u69z-wsDIc_004285": {
    "video_start": 4285,
    "video_end": 4383,
    "anomaly_start": 53,
    "anomaly_end": 69,
    "anomaly_class": "ego: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "1u69z-wsDIc_005170": {
    "video_start": 5170,
    "video_end": 5268,
    "anomaly_start": 47,
    "anomaly_end": 76,
    "anomaly_class": "ego: lateral",
    "num_frames": 99,
    "subset": "val"
  },
  "1u69z-wsDIc_005270": {
    "video_start": 5270,
    "video_end": 5347,
    "anomaly_start": 40,
    "anomaly_end": 57,
    "anomaly_class": "ego: turning",
    "num_frames": 78,
    "subset": "val"
  },
  "2SAby_5t94M_000456": {
    "video_start": 456,
    "video_end": 560,
    "anomaly_start": 36,
    "anomaly_end": 43,
    "anomaly_class": "ego: pedestrian",
    "num_frames": 105,
    "subset": "val"
  },
  "2SAby_5t94M_001122": {
    "video_start": 1122,
    "video_end": 1219,
    "anomaly_start": 24,
    "anomaly_end": 85,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 98,
    "subset": "val"
  },
  "2SAby_5t94M_001932": {
    "video_start": 1932,
    "video_end": 2019,
    "anomaly_start": 26,
    "anomaly_end": 55,
    "anomaly_class": "ego: pedestrian",
    "num_frames": 88,
    "subset": "val"
  },
  "2SAby_5t94M_002605": {
    "video_start": 2605,
    "video_end": 2720,
    "anomaly_start": 21,
    "anomaly_end": 93,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 116,
    "subset": "val"
  },
  "2TmFM9p1KF8_000141": {
    "video_start": 141,
    "video_end": 216,
    "anomaly_start": 25,
    "anomaly_end": 43,
    "anomaly_class": "ego: turning",
    "num_frames": 76,
    "subset": "val"
  },
  "2TmFM9p1KF8_000218": {
    "video_start": 218,
    "video_end": 376,
    "anomaly_start": 27,
    "anomaly_end": 63,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 159,
    "subset": "val"
  },
  "2TmFM9p1KF8_000378": {
    "video_start": 378,
    "video_end": 478,
    "anomaly_start": 43,
    "anomaly_end": 78,
    "anomaly_class": "ego: turning",
    "num_frames": 101,
    "subset": "val"
  },
  "2TmFM9p1KF8_000480": {
    "video_start": 480,
    "video_end": 599,
    "anomaly_start": 19,
    "anomaly_end": 67,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 120,
    "subset": "val"
  },
  "2TmFM9p1KF8_001489": {
    "video_start": 1489,
    "video_end": 1609,
    "anomaly_start": 22,
    "anomaly_end": 58,
    "anomaly_class": "ego: turning",
    "num_frames": 121,
    "subset": "val"
  },
  "2TmFM9p1KF8_002228": {
    "video_start": 2228,
    "video_end": 2298,
    "anomaly_start": 18,
    "anomaly_end": 58,
    "anomaly_class": "ego: turning",
    "num_frames": 71,
    "subset": "val"
  },
  "2TmFM9p1KF8_002688": {
    "video_start": 2688,
    "video_end": 2768,
    "anomaly_start": 32,
    "anomaly_end": 53,
    "anomaly_class": "ego: turning",
    "num_frames": 81,
    "subset": "val"
  },
  "2TmFM9p1KF8_003139": {
    "video_start": 3139,
    "video_end": 3218,
    "anomaly_start": 13,
    "anomaly_end": 43,
    "anomaly_class": "ego: oncoming",
    "num_frames": 80,
    "subset": "val"
  },
  "2TmFM9p1KF8_003220": {
    "video_start": 3220,
    "video_end": 3320,
    "anomaly_start": 38,
    "anomaly_end": 73,
    "anomaly_class": "ego: lateral",
    "num_frames": 101,
    "subset": "val"
  },
  "2TmFM9p1KF8_003725": {
    "video_start": 3725,
    "video_end": 3886,
    "anomaly_start": 22,
    "anomaly_end": 136,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 162,
    "subset": "val"
  },
  "2TmFM9p1KF8_004066": {
    "video_start": 4066,
    "video_end": 4213,
    "anomaly_start": 34,
    "anomaly_end": 126,
    "anomaly_class": "ego: turning",
    "num_frames": 148,
    "subset": "val"
  },
  "2TmFM9p1KF8_004447": {
    "video_start": 4447,
    "video_end": 4588,
    "anomaly_start": 40,
    "anomaly_end": 87,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 142,
    "subset": "val"
  },
  "3Sqeb-l1RPA_000905": {
    "video_start": 905,
    "video_end": 981,
    "anomaly_start": 33,
    "anomaly_end": 64,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 77,
    "subset": "val"
  },
  "3Sqeb-l1RPA_000983": {
    "video_start": 983,
    "video_end": 1058,
    "anomaly_start": 12,
    "anomaly_end": 32,
    "anomaly_class": "ego: start_stop_or_stationary",
    "num_frames": 76,
    "subset": "val"
  },
  "3Sqeb-l1RPA_001528": {
    "video_start": 1528,
    "video_end": 1649,
    "anomaly_start": 52,
    "anomaly_end": 90,
    "anomaly_class": "ego: obstacle",
    "num_frames": 122,
    "subset": "val"
  },
  "3Sqeb-l1RPA_001953": {
    "video_start": 1953,
    "video_end": 2032,
    "anomaly_start": 28,
    "anomaly_end": 56,
    "anomaly_class": "other: turning",
    "num_frames": 80,
    "subset": "val"
  },
  "3Sqeb-l1RPA_002809": {
    "video_start": 2809,
    "video_end": 2898,
    "anomaly_start": 22,
    "anomaly_end": 46,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 90,
    "subset": "val"
  },
  "3sGShQb_HwU_001196": {
    "video_start": 1196,
    "video_end": 1274,
    "anomaly_start": 47,
    "anomaly_end": 56,
    "anomaly_class": "ego: start_stop_or_stationary",
    "num_frames": 79,
    "subset": "val"
  },
  "3sGShQb_HwU_001386": {
    "video_start": 1386,
    "video_end": 1494,
    "anomaly_start": 47,
    "anomaly_end": 77,
    "anomaly_class": "other: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "3sGShQb_HwU_001496": {
    "video_start": 1496,
    "video_end": 1605,
    "anomaly_start": 56,
    "anomaly_end": 84,
    "anomaly_class": "other: turning",
    "num_frames": 110,
    "subset": "val"
  },
  "3sGShQb_HwU_001607": {
    "video_start": 1607,
    "video_end": 1704,
    "anomaly_start": 37,
    "anomaly_end": 60,
    "anomaly_class": "ego: turning",
    "num_frames": 98,
    "subset": "val"
  },
  "3sGShQb_HwU_005285": {
    "video_start": 5285,
    "video_end": 5393,
    "anomaly_start": 59,
    "anomaly_end": 79,
    "anomaly_class": "ego: oncoming",
    "num_frames": 109,
    "subset": "val"
  },
  "3sGShQb_HwU_005523": {
    "video_start": 5523,
    "video_end": 5649,
    "anomaly_start": 45,
    "anomaly_end": 86,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 127,
    "subset": "val"
  },
  "3tEZvtQZ18Q_000191": {
    "video_start": 191,
    "video_end": 299,
    "anomaly_start": 40,
    "anomaly_end": 89,
    "anomaly_class": "ego: oncoming",
    "num_frames": 109,
    "subset": "val"
  },
  "3tEZvtQZ18Q_000691": {
    "video_start": 691,
    "video_end": 789,
    "anomaly_start": 56,
    "anomaly_end": 89,
    "anomaly_class": "other: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "3tEZvtQZ18Q_000901": {
    "video_start": 901,
    "video_end": 979,
    "anomaly_start": 37,
    "anomaly_end": 56,
    "anomaly_class": "other: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "3tEZvtQZ18Q_001071": {
    "video_start": 1071,
    "video_end": 1170,
    "anomaly_start": 26,
    "anomaly_end": 84,
    "anomaly_class": "other: turning",
    "num_frames": 100,
    "subset": "val"
  },
  "3tEZvtQZ18Q_001364": {
    "video_start": 1364,
    "video_end": 1442,
    "anomaly_start": 33,
    "anomaly_end": 60,
    "anomaly_class": "other: oncoming",
    "num_frames": 79,
    "subset": "val"
  },
  "3tEZvtQZ18Q_001547": {
    "video_start": 1547,
    "video_end": 1685,
    "anomaly_start": 48,
    "anomaly_end": 89,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 139,
    "subset": "val"
  },
  "3tEZvtQZ18Q_001687": {
    "video_start": 1687,
    "video_end": 1805,
    "anomaly_start": 36,
    "anomaly_end": 119,
    "anomaly_class": "other: turning",
    "num_frames": 119,
    "subset": "val"
  },
  "3tEZvtQZ18Q_001807": {
    "video_start": 1807,
    "video_end": 1905,
    "anomaly_start": 37,
    "anomaly_end": 90,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 99,
    "subset": "val"
  },
  "3tEZvtQZ18Q_003665": {
    "video_start": 3665,
    "video_end": 3772,
    "anomaly_start": 26,
    "anomaly_end": 104,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 108,
    "subset": "val"
  },
  "3tEZvtQZ18Q_003975": {
    "video_start": 3975,
    "video_end": 4093,
    "anomaly_start": 44,
    "anomaly_end": 83,
    "anomaly_class": "other: pedestrian",
    "num_frames": 119,
    "subset": "val"
  },
  "3tEZvtQZ18Q_004095": {
    "video_start": 4095,
    "video_end": 4195,
    "anomaly_start": 46,
    "anomaly_end": 82,
    "anomaly_class": "other: pedestrian",
    "num_frames": 101,
    "subset": "val"
  },
  "3tEZvtQZ18Q_004470": {
    "video_start": 4470,
    "video_end": 4638,
    "anomaly_start": 27,
    "anomaly_end": 66,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 169,
    "subset": "val"
  },
  "3tEZvtQZ18Q_004890": {
    "video_start": 4890,
    "video_end": 4968,
    "anomaly_start": 52,
    "anomaly_end": 79,
    "anomaly_class": "ego: start_stop_or_stationary",
    "num_frames": 79,
    "subset": "val"
  },
  "3tEZvtQZ18Q_005180": {
    "video_start": 5180,
    "video_end": 5268,
    "anomaly_start": 37,
    "anomaly_end": 65,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "3tEZvtQZ18Q_005536": {
    "video_start": 5536,
    "video_end": 5589,
    "anomaly_start": 47,
    "anomaly_end": 54,
    "anomaly_class": "ego: turning",
    "num_frames": 54,
    "subset": "val"
  },
  "3tEZvtQZ18Q_005646": {
    "video_start": 5646,
    "video_end": 5764,
    "anomaly_start": 33,
    "anomaly_end": 83,
    "anomaly_class": "ego: oncoming",
    "num_frames": 119,
    "subset": "val"
  },
  "3u_CIo9IaWo_000500": {
    "video_start": 500,
    "video_end": 578,
    "anomaly_start": 38,
    "anomaly_end": 60,
    "anomaly_class": "ego: oncoming",
    "num_frames": 79,
    "subset": "val"
  },
  "3u_CIo9IaWo_000782": {
    "video_start": 782,
    "video_end": 872,
    "anomaly_start": 33,
    "anomaly_end": 88,
    "anomaly_class": "ego: turning",
    "num_frames": 91,
    "subset": "val"
  },
  "3u_CIo9IaWo_002249": {
    "video_start": 2249,
    "video_end": 2353,
    "anomaly_start": 41,
    "anomaly_end": 82,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 105,
    "subset": "val"
  },
  "3u_CIo9IaWo_002355": {
    "video_start": 2355,
    "video_end": 2456,
    "anomaly_start": 30,
    "anomaly_end": 102,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 102,
    "subset": "val"
  },
  "3u_CIo9IaWo_002588": {
    "video_start": 2588,
    "video_end": 2690,
    "anomaly_start": 60,
    "anomaly_end": 82,
    "anomaly_class": "ego: oncoming",
    "num_frames": 103,
    "subset": "val"
  },
  "3u_CIo9IaWo_002793": {
    "video_start": 2793,
    "video_end": 2891,
    "anomaly_start": 39,
    "anomaly_end": 96,
    "anomaly_class": "ego: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "3u_CIo9IaWo_004998": {
    "video_start": 4998,
    "video_end": 5091,
    "anomaly_start": 84,
    "anomaly_end": 94,
    "anomaly_class": "ego: turning",
    "num_frames": 94,
    "subset": "val"
  },
  "4K_6s1n6BpU_000319": {
    "video_start": 319,
    "video_end": 417,
    "anomaly_start": 14,
    "anomaly_end": 56,
    "anomaly_class": "ego: lateral",
    "num_frames": 99,
    "subset": "val"
  },
  "4K_6s1n6BpU_000804": {
    "video_start": 804,
    "video_end": 918,
    "anomaly_start": 42,
    "anomaly_end": 66,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 115,
    "subset": "val"
  },
  "4K_6s1n6BpU_001222": {
    "video_start": 1222,
    "video_end": 1342,
    "anomaly_start": 37,
    "anomaly_end": 61,
    "anomaly_class": "other: obstacle",
    "num_frames": 121,
    "subset": "val"
  },
  "4K_6s1n6BpU_001530": {
    "video_start": 1530,
    "video_end": 1674,
    "anomaly_start": 34,
    "anomaly_end": 50,
    "anomaly_class": "ego: turning",
    "num_frames": 145,
    "subset": "val"
  },
  "4K_6s1n6BpU_001997": {
    "video_start": 1997,
    "video_end": 2096,
    "anomaly_start": 34,
    "anomaly_end": 68,
    "anomaly_class": "ego: turning",
    "num_frames": 100,
    "subset": "val"
  },
  "4K_6s1n6BpU_002431": {
    "video_start": 2431,
    "video_end": 2568,
    "anomaly_start": 11,
    "anomaly_end": 78,
    "anomaly_class": "ego: oncoming",
    "num_frames": 138,
    "subset": "val"
  },
  "4K_6s1n6BpU_002788": {
    "video_start": 2788,
    "video_end": 2866,
    "anomaly_start": 36,
    "anomaly_end": 66,
    "anomaly_class": "ego: lateral",
    "num_frames": 79,
    "subset": "val"
  },
  "4K_6s1n6BpU_004403": {
    "video_start": 4403,
    "video_end": 4502,
    "anomaly_start": 28,
    "anomaly_end": 60,
    "anomaly_class": "ego: lateral",
    "num_frames": 100,
    "subset": "val"
  },
  "4K_6s1n6BpU_004597": {
    "video_start": 4597,
    "video_end": 4677,
    "anomaly_start": 25,
    "anomaly_end": 59,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 81,
    "subset": "val"
  },
  "4K_6s1n6BpU_005656": {
    "video_start": 5656,
    "video_end": 5824,
    "anomaly_start": 20,
    "anomaly_end": 157,
    "anomaly_class": "other: lateral",
    "num_frames": 169,
    "subset": "val"
  },
  "4OGV0AbV91U_000667": {
    "video_start": 667,
    "video_end": 775,
    "anomaly_start": 1,
    "anomaly_end": 73,
    "anomaly_class": "other: unknown",
    "num_frames": 109,
    "subset": "val"
  },
  "4OGV0AbV91U_001201": {
    "video_start": 1201,
    "video_end": 1274,
    "anomaly_start": 32,
    "anomaly_end": 53,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 74,
    "subset": "val"
  },
  "4OGV0AbV91U_001276": {
    "video_start": 1276,
    "video_end": 1378,
    "anomaly_start": 21,
    "anomaly_end": 73,
    "anomaly_class": "ego: lateral",
    "num_frames": 103,
    "subset": "val"
  },
  "4OGV0AbV91U_001988": {
    "video_start": 1988,
    "video_end": 2107,
    "anomaly_start": 32,
    "anomaly_end": 73,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 120,
    "subset": "val"
  },
  "4OGV0AbV91U_003527": {
    "video_start": 3527,
    "video_end": 3610,
    "anomaly_start": 41,
    "anomaly_end": 69,
    "anomaly_class": "other: turning",
    "num_frames": 84,
    "subset": "val"
  },
  "4OGV0AbV91U_003700": {
    "video_start": 3700,
    "video_end": 3795,
    "anomaly_start": 42,
    "anomaly_end": 57,
    "anomaly_class": "ego: pedestrian",
    "num_frames": 96,
    "subset": "val"
  },
  "4OGV0AbV91U_004279": {
    "video_start": 4279,
    "video_end": 4350,
    "anomaly_start": 18,
    "anomaly_end": 40,
    "anomaly_class": "other: turning",
    "num_frames": 72,
    "subset": "val"
  },
  "4OGV0AbV91U_004435": {
    "video_start": 4435,
    "video_end": 4536,
    "anomaly_start": 17,
    "anomaly_end": 43,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 102,
    "subset": "val"
  },
  "4OGV0AbV91U_004538": {
    "video_start": 4538,
    "video_end": 4610,
    "anomaly_start": 26,
    "anomaly_end": 47,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 73,
    "subset": "val"
  },
  "4OGV0AbV91U_004771": {
    "video_start": 4771,
    "video_end": 4844,
    "anomaly_start": 12,
    "anomaly_end": 46,
    "anomaly_class": "other: lateral",
    "num_frames": 74,
    "subset": "val"
  },
  "4wKjxDXnmYs_000061": {
    "video_start": 61,
    "video_end": 115,
    "anomaly_start": 10,
    "anomaly_end": 36,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 55,
    "subset": "val"
  },
  "4wKjxDXnmYs_000311": {
    "video_start": 311,
    "video_end": 385,
    "anomaly_start": 33,
    "anomaly_end": 55,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 75,
    "subset": "val"
  },
  "4wKjxDXnmYs_001051": {
    "video_start": 1051,
    "video_end": 1111,
    "anomaly_start": 28,
    "anomaly_end": 59,
    "anomaly_class": "other: turning",
    "num_frames": 61,
    "subset": "val"
  },
  "4wKjxDXnmYs_001254": {
    "video_start": 1254,
    "video_end": 1349,
    "anomaly_start": 32,
    "anomaly_end": 81,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 96,
    "subset": "val"
  },
  "4wKjxDXnmYs_001662": {
    "video_start": 1662,
    "video_end": 1750,
    "anomaly_start": 25,
    "anomaly_end": 52,
    "anomaly_class": "ego: lateral",
    "num_frames": 89,
    "subset": "val"
  },
  "4wKjxDXnmYs_002899": {
    "video_start": 2899,
    "video_end": 2976,
    "anomaly_start": 26,
    "anomaly_end": 56,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 78,
    "subset": "val"
  },
  "4wKjxDXnmYs_003500": {
    "video_start": 3500,
    "video_end": 3577,
    "anomaly_start": 24,
    "anomaly_end": 52,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 78,
    "subset": "val"
  },
  "4wKjxDXnmYs_003579": {
    "video_start": 3579,
    "video_end": 3645,
    "anomaly_start": 30,
    "anomaly_end": 52,
    "anomaly_class": "ego: lateral",
    "num_frames": 67,
    "subset": "val"
  },
  "4wKjxDXnmYs_003798": {
    "video_start": 3798,
    "video_end": 3886,
    "anomaly_start": 22,
    "anomaly_end": 63,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 89,
    "subset": "val"
  },
  "4wKjxDXnmYs_003888": {
    "video_start": 3888,
    "video_end": 3967,
    "anomaly_start": 27,
    "anomaly_end": 56,
    "anomaly_class": "ego: lateral",
    "num_frames": 80,
    "subset": "val"
  },
  "4wKjxDXnmYs_003969": {
    "video_start": 3969,
    "video_end": 4026,
    "anomaly_start": 10,
    "anomaly_end": 47,
    "anomaly_class": "other: lateral",
    "num_frames": 58,
    "subset": "val"
  },
  "4wKjxDXnmYs_004028": {
    "video_start": 4028,
    "video_end": 4106,
    "anomaly_start": 23,
    "anomaly_end": 41,
    "anomaly_class": "ego: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "4wKjxDXnmYs_004457": {
    "video_start": 4457,
    "video_end": 4521,
    "anomaly_start": 40,
    "anomaly_end": 57,
    "anomaly_class": "ego: lateral",
    "num_frames": 65,
    "subset": "val"
  },
  "4wKjxDXnmYs_005030": {
    "video_start": 5030,
    "video_end": 5096,
    "anomaly_start": 28,
    "anomaly_end": 51,
    "anomaly_class": "other: lateral",
    "num_frames": 67,
    "subset": "val"
  },
  "4wKjxDXnmYs_005161": {
    "video_start": 5161,
    "video_end": 5213,
    "anomaly_start": 16,
    "anomaly_end": 38,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 53,
    "subset": "val"
  },
  "4wKjxDXnmYs_005595": {
    "video_start": 5595,
    "video_end": 5662,
    "anomaly_start": 34,
    "anomaly_end": 66,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 68,
    "subset": "val"
  },
  "5vKPYV5w6pw_000181": {
    "video_start": 181,
    "video_end": 293,
    "anomaly_start": 57,
    "anomaly_end": 76,
    "anomaly_class": "ego: turning",
    "num_frames": 113,
    "subset": "val"
  },
  "5vKPYV5w6pw_000623": {
    "video_start": 623,
    "video_end": 701,
    "anomaly_start": 34,
    "anomaly_end": 48,
    "anomaly_class": "other: oncoming",
    "num_frames": 79,
    "subset": "val"
  },
  "5vKPYV5w6pw_000703": {
    "video_start": 703,
    "video_end": 771,
    "anomaly_start": 37,
    "anomaly_end": 52,
    "anomaly_class": "other: oncoming",
    "num_frames": 69,
    "subset": "val"
  },
  "5vKPYV5w6pw_000874": {
    "video_start": 874,
    "video_end": 932,
    "anomaly_start": 27,
    "anomaly_end": 42,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 59,
    "subset": "val"
  },
  "5vKPYV5w6pw_003867": {
    "video_start": 3867,
    "video_end": 4005,
    "anomaly_start": 35,
    "anomaly_end": 62,
    "anomaly_class": "other: oncoming",
    "num_frames": 139,
    "subset": "val"
  },
  "5vKPYV5w6pw_004134": {
    "video_start": 4134,
    "video_end": 4222,
    "anomaly_start": 41,
    "anomaly_end": 62,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "5vKPYV5w6pw_004515": {
    "video_start": 4515,
    "video_end": 4613,
    "anomaly_start": 42,
    "anomaly_end": 64,
    "anomaly_class": "other: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "5vKPYV5w6pw_005297": {
    "video_start": 5297,
    "video_end": 5395,
    "anomaly_start": 57,
    "anomaly_end": 76,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 99,
    "subset": "val"
  },
  "6E2N9ld0eHg_000180": {
    "video_start": 180,
    "video_end": 288,
    "anomaly_start": 59,
    "anomaly_end": 79,
    "anomaly_class": "ego: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "6E2N9ld0eHg_000510": {
    "video_start": 510,
    "video_end": 609,
    "anomaly_start": 58,
    "anomaly_end": 82,
    "anomaly_class": "ego: turning",
    "num_frames": 100,
    "subset": "val"
  },
  "6E2N9ld0eHg_001173": {
    "video_start": 1173,
    "video_end": 1261,
    "anomaly_start": 53,
    "anomaly_end": 70,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "6E2N9ld0eHg_001263": {
    "video_start": 1263,
    "video_end": 1371,
    "anomaly_start": 46,
    "anomaly_end": 108,
    "anomaly_class": "ego: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "6E2N9ld0eHg_001813": {
    "video_start": 1813,
    "video_end": 1915,
    "anomaly_start": 42,
    "anomaly_end": 67,
    "anomaly_class": "ego: turning",
    "num_frames": 103,
    "subset": "val"
  },
  "6E2N9ld0eHg_002998": {
    "video_start": 2998,
    "video_end": 3086,
    "anomaly_start": 47,
    "anomaly_end": 65,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 89,
    "subset": "val"
  },
  "6E2N9ld0eHg_003418": {
    "video_start": 3418,
    "video_end": 3506,
    "anomaly_start": 45,
    "anomaly_end": 65,
    "anomaly_class": "other: oncoming",
    "num_frames": 89,
    "subset": "val"
  },
  "6E2N9ld0eHg_004179": {
    "video_start": 4179,
    "video_end": 4287,
    "anomaly_start": 27,
    "anomaly_end": 80,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 109,
    "subset": "val"
  },
  "6E2N9ld0eHg_004669": {
    "video_start": 4669,
    "video_end": 4807,
    "anomaly_start": 36,
    "anomaly_end": 59,
    "anomaly_class": "other: pedestrian",
    "num_frames": 139,
    "subset": "val"
  },
  "6E2N9ld0eHg_005717": {
    "video_start": 5717,
    "video_end": 5874,
    "anomaly_start": 47,
    "anomaly_end": 115,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 158,
    "subset": "val"
  },
  "6I7Dz0qF4NA_000869": {
    "video_start": 869,
    "video_end": 968,
    "anomaly_start": 20,
    "anomaly_end": 52,
    "anomaly_class": "other: turning",
    "num_frames": 100,
    "subset": "val"
  },
  "7uTMZ5-aWKA_000490": {
    "video_start": 490,
    "video_end": 588,
    "anomaly_start": 31,
    "anomaly_end": 85,
    "anomaly_class": "ego: oncoming",
    "num_frames": 99,
    "subset": "val"
  },
  "7uTMZ5-aWKA_001080": {
    "video_start": 1080,
    "video_end": 1189,
    "anomaly_start": 31,
    "anomaly_end": 79,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 110,
    "subset": "val"
  },
  "7uTMZ5-aWKA_001191": {
    "video_start": 1191,
    "video_end": 1293,
    "anomaly_start": 51,
    "anomaly_end": 91,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 103,
    "subset": "val"
  },
  "7uTMZ5-aWKA_001505": {
    "video_start": 1505,
    "video_end": 1603,
    "anomaly_start": 42,
    "anomaly_end": 80,
    "anomaly_class": "ego: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "7uTMZ5-aWKA_001695": {
    "video_start": 1695,
    "video_end": 1826,
    "anomaly_start": 28,
    "anomaly_end": 102,
    "anomaly_class": "ego: oncoming",
    "num_frames": 132,
    "subset": "val"
  },
  "7uTMZ5-aWKA_001975": {
    "video_start": 1975,
    "video_end": 2113,
    "anomaly_start": 62,
    "anomaly_end": 112,
    "anomaly_class": "ego: lateral",
    "num_frames": 139,
    "subset": "val"
  },
  "7uTMZ5-aWKA_002115": {
    "video_start": 2115,
    "video_end": 2265,
    "anomaly_start": 20,
    "anomaly_end": 51,
    "anomaly_class": "other: turning",
    "num_frames": 151,
    "subset": "val"
  },
  "7uTMZ5-aWKA_002982": {
    "video_start": 2982,
    "video_end": 3178,
    "anomaly_start": 109,
    "anomaly_end": 145,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 197,
    "subset": "val"
  },
  "7uTMZ5-aWKA_003948": {
    "video_start": 3948,
    "video_end": 4075,
    "anomaly_start": 32,
    "anomaly_end": 92,
    "anomaly_class": "ego: pedestrian",
    "num_frames": 128,
    "subset": "val"
  },
  "88LcRU7uEFE_000073": {
    "video_start": 73,
    "video_end": 198,
    "anomaly_start": 38,
    "anomaly_end": 83,
    "anomaly_class": "ego: lateral",
    "num_frames": 126,
    "subset": "val"
  },
  "88LcRU7uEFE_000511": {
    "video_start": 511,
    "video_end": 610,
    "anomaly_start": 33,
    "anomaly_end": 71,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 100,
    "subset": "val"
  },
  "88LcRU7uEFE_001137": {
    "video_start": 1137,
    "video_end": 1236,
    "anomaly_start": 38,
    "anomaly_end": 89,
    "anomaly_class": "ego: turning",
    "num_frames": 100,
    "subset": "val"
  },
  "88LcRU7uEFE_001468": {
    "video_start": 1468,
    "video_end": 1586,
    "anomaly_start": 47,
    "anomaly_end": 106,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 119,
    "subset": "val"
  },
  "88LcRU7uEFE_002275": {
    "video_start": 2275,
    "video_end": 2423,
    "anomaly_start": 53,
    "anomaly_end": 104,
    "anomaly_class": "ego: unknown",
    "num_frames": 149,
    "subset": "val"
  },
  "88LcRU7uEFE_002882": {
    "video_start": 2882,
    "video_end": 2930,
    "anomaly_start": 41,
    "anomaly_end": 49,
    "anomaly_class": "ego: oncoming",
    "num_frames": 49,
    "subset": "val"
  },
  "88LcRU7uEFE_004261": {
    "video_start": 4261,
    "video_end": 4369,
    "anomaly_start": 45,
    "anomaly_end": 71,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 109,
    "subset": "val"
  },
  "8dI7OolIEXY_000286": {
    "video_start": 286,
    "video_end": 417,
    "anomaly_start": 63,
    "anomaly_end": 101,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 132,
    "subset": "val"
  },
  "8dI7OolIEXY_000708": {
    "video_start": 708,
    "video_end": 793,
    "anomaly_start": 30,
    "anomaly_end": 82,
    "anomaly_class": "ego: lateral",
    "num_frames": 86,
    "subset": "val"
  },
  "8dI7OolIEXY_000880": {
    "video_start": 880,
    "video_end": 983,
    "anomaly_start": 33,
    "anomaly_end": 52,
    "anomaly_class": "ego: lateral",
    "num_frames": 104,
    "subset": "val"
  },
  "8dI7OolIEXY_001371": {
    "video_start": 1371,
    "video_end": 1459,
    "anomaly_start": 21,
    "anomaly_end": 66,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "8dI7OolIEXY_004818": {
    "video_start": 4818,
    "video_end": 4925,
    "anomaly_start": 20,
    "anomaly_end": 103,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 108,
    "subset": "val"
  },
  "8dI7OolIEXY_005013": {
    "video_start": 5013,
    "video_end": 5105,
    "anomaly_start": 16,
    "anomaly_end": 76,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 93,
    "subset": "val"
  },
  "8dI7OolIEXY_005194": {
    "video_start": 5194,
    "video_end": 5287,
    "anomaly_start": 52,
    "anomaly_end": 71,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 94,
    "subset": "val"
  },
  "8dI7OolIEXY_005681": {
    "video_start": 5681,
    "video_end": 5839,
    "anomaly_start": 43,
    "anomaly_end": 76,
    "anomaly_class": "other: turning",
    "num_frames": 159,
    "subset": "val"
  },
  "8dI7OolIEXY_006141": {
    "video_start": 6141,
    "video_end": 6218,
    "anomaly_start": 43,
    "anomaly_end": 61,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 78,
    "subset": "val"
  },
  "8dI7OolIEXY_006288": {
    "video_start": 6288,
    "video_end": 6354,
    "anomaly_start": 7,
    "anomaly_end": 47,
    "anomaly_class": "other: lateral",
    "num_frames": 67,
    "subset": "val"
  },
  "90gyBengKDs_000062": {
    "video_start": 62,
    "video_end": 210,
    "anomaly_start": 50,
    "anomaly_end": 78,
    "anomaly_class": "other: unknown",
    "num_frames": 149,
    "subset": "val"
  },
  "90gyBengKDs_000372": {
    "video_start": 372,
    "video_end": 514,
    "anomaly_start": 39,
    "anomaly_end": 69,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 143,
    "subset": "val"
  },
  "90gyBengKDs_000752": {
    "video_start": 752,
    "video_end": 838,
    "anomaly_start": 21,
    "anomaly_end": 78,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 87,
    "subset": "val"
  },
  "90gyBengKDs_001133": {
    "video_start": 1133,
    "video_end": 1204,
    "anomaly_start": 37,
    "anomaly_end": 55,
    "anomaly_class": "ego: turning",
    "num_frames": 72,
    "subset": "val"
  },
  "90gyBengKDs_002709": {
    "video_start": 2709,
    "video_end": 2785,
    "anomaly_start": 18,
    "anomaly_end": 72,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 77,
    "subset": "val"
  },
  "90gyBengKDs_003658": {
    "video_start": 3658,
    "video_end": 3760,
    "anomaly_start": 33,
    "anomaly_end": 34,
    "anomaly_class": "ego: turning",
    "num_frames": 103,
    "subset": "val"
  },
  "90gyBengKDs_004112": {
    "video_start": 4112,
    "video_end": 4189,
    "anomaly_start": 12,
    "anomaly_end": 40,
    "anomaly_class": "other: lateral",
    "num_frames": 78,
    "subset": "val"
  },
  "90gyBengKDs_004191": {
    "video_start": 4191,
    "video_end": 4231,
    "anomaly_start": 20,
    "anomaly_end": 41,
    "anomaly_class": "ego: oncoming",
    "num_frames": 41,
    "subset": "val"
  },
  "90gyBengKDs_004250": {
    "video_start": 4250,
    "video_end": 4310,
    "anomaly_start": 40,
    "anomaly_end": 59,
    "anomaly_class": "ego: turning",
    "num_frames": 61,
    "subset": "val"
  },
  "9Q0FCoNxL8I_000073": {
    "video_start": 73,
    "video_end": 219,
    "anomaly_start": 39,
    "anomaly_end": 147,
    "anomaly_class": "ego: oncoming",
    "num_frames": 147,
    "subset": "val"
  },
  "9Q0FCoNxL8I_000676": {
    "video_start": 676,
    "video_end": 784,
    "anomaly_start": 29,
    "anomaly_end": 84,
    "anomaly_class": "other: lateral",
    "num_frames": 109,
    "subset": "val"
  },
  "9Q0FCoNxL8I_001441": {
    "video_start": 1441,
    "video_end": 1550,
    "anomaly_start": 31,
    "anomaly_end": 97,
    "anomaly_class": "ego: oncoming",
    "num_frames": 110,
    "subset": "val"
  },
  "9Q0FCoNxL8I_002076": {
    "video_start": 2076,
    "video_end": 2215,
    "anomaly_start": 19,
    "anomaly_end": 60,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 140,
    "subset": "val"
  },
  "9Q0FCoNxL8I_003218": {
    "video_start": 3218,
    "video_end": 3336,
    "anomaly_start": 50,
    "anomaly_end": 105,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 119,
    "subset": "val"
  },
  "9Q0FCoNxL8I_003338": {
    "video_start": 3338,
    "video_end": 3438,
    "anomaly_start": 29,
    "anomaly_end": 61,
    "anomaly_class": "other: pedestrian",
    "num_frames": 101,
    "subset": "val"
  },
  "9Q0FCoNxL8I_003545": {
    "video_start": 3545,
    "video_end": 3653,
    "anomaly_start": 28,
    "anomaly_end": 72,
    "anomaly_class": "other: pedestrian",
    "num_frames": 109,
    "subset": "val"
  },
  "9Q0FCoNxL8I_004096": {
    "video_start": 4096,
    "video_end": 4217,
    "anomaly_start": 37,
    "anomaly_end": 96,
    "anomaly_class": "other: turning",
    "num_frames": 122,
    "subset": "val"
  },
  "9Q0FCoNxL8I_004219": {
    "video_start": 4219,
    "video_end": 4358,
    "anomaly_start": 14,
    "anomaly_end": 92,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 140,
    "subset": "val"
  },
  "9Q0FCoNxL8I_004705": {
    "video_start": 4705,
    "video_end": 4802,
    "anomaly_start": 34,
    "anomaly_end": 98,
    "anomaly_class": "ego: turning",
    "num_frames": 98,
    "subset": "val"
  },
  "9Q0FCoNxL8I_006055": {
    "video_start": 6055,
    "video_end": 6218,
    "anomaly_start": 104,
    "anomaly_end": 134,
    "anomaly_class": "ego: turning",
    "num_frames": 164,
    "subset": "val"
  },
  "CyeT8rEQOpQ_000185": {
    "video_start": 185,
    "video_end": 343,
    "anomaly_start": 61,
    "anomaly_end": 94,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 159,
    "subset": "val"
  },
  "CyeT8rEQOpQ_000957": {
    "video_start": 957,
    "video_end": 1008,
    "anomaly_start": 23,
    "anomaly_end": 45,
    "anomaly_class": "ego: lateral",
    "num_frames": 52,
    "subset": "val"
  },
  "CyeT8rEQOpQ_001106": {
    "video_start": 1106,
    "video_end": 1228,
    "anomaly_start": 64,
    "anomaly_end": 88,
    "anomaly_class": "other: lateral",
    "num_frames": 123,
    "subset": "val"
  },
  "CyeT8rEQOpQ_001407": {
    "video_start": 1407,
    "video_end": 1521,
    "anomaly_start": 53,
    "anomaly_end": 90,
    "anomaly_class": "ego: lateral",
    "num_frames": 115,
    "subset": "val"
  },
  "CyeT8rEQOpQ_002190": {
    "video_start": 2190,
    "video_end": 2309,
    "anomaly_start": 44,
    "anomaly_end": 102,
    "anomaly_class": "ego: unknown",
    "num_frames": 120,
    "subset": "val"
  },
  "CyeT8rEQOpQ_003424": {
    "video_start": 3424,
    "video_end": 3516,
    "anomaly_start": 41,
    "anomaly_end": 55,
    "anomaly_class": "ego: lateral",
    "num_frames": 93,
    "subset": "val"
  },
  "CyeT8rEQOpQ_004542": {
    "video_start": 4542,
    "video_end": 4635,
    "anomaly_start": 32,
    "anomaly_end": 78,
    "anomaly_class": "ego: oncoming",
    "num_frames": 94,
    "subset": "val"
  },
  "CyeT8rEQOpQ_005848": {
    "video_start": 5848,
    "video_end": 5954,
    "anomaly_start": 41,
    "anomaly_end": 69,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 107,
    "subset": "val"
  },
  "DSBGS6aRzgc_000561": {
    "video_start": 561,
    "video_end": 659,
    "anomaly_start": 52,
    "anomaly_end": 73,
    "anomaly_class": "ego: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "DSBGS6aRzgc_000851": {
    "video_start": 851,
    "video_end": 949,
    "anomaly_start": 41,
    "anomaly_end": 79,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 99,
    "subset": "val"
  },
  "DSBGS6aRzgc_001418": {
    "video_start": 1418,
    "video_end": 1530,
    "anomaly_start": 64,
    "anomaly_end": 113,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 113,
    "subset": "val"
  },
  "DSBGS6aRzgc_002793": {
    "video_start": 2793,
    "video_end": 2872,
    "anomaly_start": 39,
    "anomaly_end": 53,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 80,
    "subset": "val"
  },
  "DSBGS6aRzgc_002874": {
    "video_start": 2874,
    "video_end": 2972,
    "anomaly_start": 23,
    "anomaly_end": 61,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 99,
    "subset": "val"
  },
  "DSBGS6aRzgc_005264": {
    "video_start": 5264,
    "video_end": 5362,
    "anomaly_start": 27,
    "anomaly_end": 81,
    "anomaly_class": "other: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "DSBGS6aRzgc_005364": {
    "video_start": 5364,
    "video_end": 5452,
    "anomaly_start": 38,
    "anomaly_end": 67,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 89,
    "subset": "val"
  },
  "DSBGS6aRzgc_005564": {
    "video_start": 5564,
    "video_end": 5651,
    "anomaly_start": 45,
    "anomaly_end": 86,
    "anomaly_class": "ego: start_stop_or_stationary",
    "num_frames": 88,
    "subset": "val"
  },
  "D_pyFV4nKd4_000211": {
    "video_start": 211,
    "video_end": 327,
    "anomaly_start": 57,
    "anomaly_end": 116,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 117,
    "subset": "val"
  },
  "D_pyFV4nKd4_000329": {
    "video_start": 329,
    "video_end": 434,
    "anomaly_start": 39,
    "anomaly_end": 75,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 106,
    "subset": "val"
  },
  "D_pyFV4nKd4_000776": {
    "video_start": 776,
    "video_end": 876,
    "anomaly_start": 50,
    "anomaly_end": 74,
    "anomaly_class": "other: turning",
    "num_frames": 101,
    "subset": "val"
  },
  "D_pyFV4nKd4_000878": {
    "video_start": 878,
    "video_end": 977,
    "anomaly_start": 32,
    "anomaly_end": 72,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 100,
    "subset": "val"
  },
  "D_pyFV4nKd4_001555": {
    "video_start": 1555,
    "video_end": 1663,
    "anomaly_start": 51,
    "anomaly_end": 109,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 109,
    "subset": "val"
  },
  "D_pyFV4nKd4_001836": {
    "video_start": 1836,
    "video_end": 1954,
    "anomaly_start": 32,
    "anomaly_end": 64,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 119,
    "subset": "val"
  },
  "D_pyFV4nKd4_001956": {
    "video_start": 1956,
    "video_end": 2094,
    "anomaly_start": 34,
    "anomaly_end": 116,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 139,
    "subset": "val"
  },
  "D_pyFV4nKd4_002096": {
    "video_start": 2096,
    "video_end": 2194,
    "anomaly_start": 32,
    "anomaly_end": 61,
    "anomaly_class": "other: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "D_pyFV4nKd4_002653": {
    "video_start": 2653,
    "video_end": 2731,
    "anomaly_start": 36,
    "anomaly_end": 51,
    "anomaly_class": "ego: oncoming",
    "num_frames": 79,
    "subset": "val"
  },
  "D_pyFV4nKd4_002874": {
    "video_start": 2874,
    "video_end": 2972,
    "anomaly_start": 37,
    "anomaly_end": 65,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 99,
    "subset": "val"
  },
  "D_pyFV4nKd4_003492": {
    "video_start": 3492,
    "video_end": 3571,
    "anomaly_start": 36,
    "anomaly_end": 65,
    "anomaly_class": "ego: turning",
    "num_frames": 80,
    "subset": "val"
  },
  "D_pyFV4nKd4_003573": {
    "video_start": 3573,
    "video_end": 3673,
    "anomaly_start": 16,
    "anomaly_end": 84,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 101,
    "subset": "val"
  },
  "D_pyFV4nKd4_003675": {
    "video_start": 3675,
    "video_end": 3802,
    "anomaly_start": 33,
    "anomaly_end": 103,
    "anomaly_class": "ego: turning",
    "num_frames": 128,
    "subset": "val"
  },
  "D_pyFV4nKd4_004350": {
    "video_start": 4350,
    "video_end": 4428,
    "anomaly_start": 37,
    "anomaly_end": 79,
    "anomaly_class": "ego: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "D_pyFV4nKd4_004630": {
    "video_start": 4630,
    "video_end": 4708,
    "anomaly_start": 37,
    "anomaly_end": 53,
    "anomaly_class": "ego: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "DpQBo4gb6Lw_000490": {
    "video_start": 490,
    "video_end": 588,
    "anomaly_start": 38,
    "anomaly_end": 70,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 99,
    "subset": "val"
  },
  "DpQBo4gb6Lw_000590": {
    "video_start": 590,
    "video_end": 685,
    "anomaly_start": 36,
    "anomaly_end": 71,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 96,
    "subset": "val"
  },
  "DpQBo4gb6Lw_001175": {
    "video_start": 1175,
    "video_end": 1273,
    "anomaly_start": 36,
    "anomaly_end": 74,
    "anomaly_class": "other: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "DpQBo4gb6Lw_001365": {
    "video_start": 1365,
    "video_end": 1473,
    "anomaly_start": 32,
    "anomaly_end": 63,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 109,
    "subset": "val"
  },
  "DpQBo4gb6Lw_002972": {
    "video_start": 2972,
    "video_end": 3132,
    "anomaly_start": 38,
    "anomaly_end": 71,
    "anomaly_class": "other: unknown",
    "num_frames": 161,
    "subset": "val"
  },
  "DpQBo4gb6Lw_003738": {
    "video_start": 3738,
    "video_end": 3837,
    "anomaly_start": 25,
    "anomaly_end": 83,
    "anomaly_class": "other: turning",
    "num_frames": 100,
    "subset": "val"
  },
  "DyzL2sahobA_000621": {
    "video_start": 621,
    "video_end": 749,
    "anomaly_start": 42,
    "anomaly_end": 68,
    "anomaly_class": "other: lateral",
    "num_frames": 129,
    "subset": "val"
  },
  "DyzL2sahobA_002611": {
    "video_start": 2611,
    "video_end": 2860,
    "anomaly_start": 25,
    "anomaly_end": 51,
    "anomaly_class": "other: turning",
    "num_frames": 250,
    "subset": "val"
  },
  "DyzL2sahobA_004003": {
    "video_start": 4003,
    "video_end": 4128,
    "anomaly_start": 25,
    "anomaly_end": 48,
    "anomaly_class": "other: turning",
    "num_frames": 126,
    "subset": "val"
  },
  "DyzL2sahobA_004649": {
    "video_start": 4649,
    "video_end": 4717,
    "anomaly_start": 40,
    "anomaly_end": 62,
    "anomaly_class": "ego: oncoming",
    "num_frames": 69,
    "subset": "val"
  },
  "DyzL2sahobA_004719": {
    "video_start": 4719,
    "video_end": 4807,
    "anomaly_start": 28,
    "anomaly_end": 79,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "DyzL2sahobA_005120": {
    "video_start": 5120,
    "video_end": 5239,
    "anomaly_start": 36,
    "anomaly_end": 63,
    "anomaly_class": "ego: start_stop_or_stationary",
    "num_frames": 120,
    "subset": "val"
  },
  "DyzL2sahobA_005780": {
    "video_start": 5780,
    "video_end": 5876,
    "anomaly_start": 36,
    "anomaly_end": 57,
    "anomaly_class": "other: pedestrian",
    "num_frames": 97,
    "subset": "val"
  },
  "EJ4VM9wLNXQ_001699": {
    "video_start": 1699,
    "video_end": 1749,
    "anomaly_start": 41,
    "anomaly_end": 51,
    "anomaly_class": "ego: oncoming",
    "num_frames": 51,
    "subset": "val"
  },
  "EJ4VM9wLNXQ_001770": {
    "video_start": 1770,
    "video_end": 1844,
    "anomaly_start": 10,
    "anomaly_end": 75,
    "anomaly_class": "other: obstacle",
    "num_frames": 75,
    "subset": "val"
  },
  "EY8x-fyQkbk_000573": {
    "video_start": 573,
    "video_end": 731,
    "anomaly_start": 9,
    "anomaly_end": 45,
    "anomaly_class": "other: start_stop_or_stationary",
    "num_frames": 159,
    "subset": "val"
  },
  "EY8x-fyQkbk_001227": {
    "video_start": 1227,
    "video_end": 1364,
    "anomaly_start": 97,
    "anomaly_end": 123,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 138,
    "subset": "val"
  },
  "EY8x-fyQkbk_001645": {
    "video_start": 1645,
    "video_end": 1773,
    "anomaly_start": 36,
    "anomaly_end": 66,
    "anomaly_class": "ego: oncoming",
    "num_frames": 129,
    "subset": "val"
  },
  "EY8x-fyQkbk_001775": {
    "video_start": 1775,
    "video_end": 1912,
    "anomaly_start": 35,
    "anomaly_end": 69,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 138,
    "subset": "val"
  },
  "EY8x-fyQkbk_002021": {
    "video_start": 2021,
    "video_end": 2140,
    "anomaly_start": 30,
    "anomaly_end": 63,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 120,
    "subset": "val"
  },
  "EY8x-fyQkbk_002142": {
    "video_start": 2142,
    "video_end": 2252,
    "anomaly_start": 34,
    "anomaly_end": 66,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 111,
    "subset": "val"
  },
  "EY8x-fyQkbk_002649": {
    "video_start": 2649,
    "video_end": 2760,
    "anomaly_start": 39,
    "anomaly_end": 82,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 112,
    "subset": "val"
  },
  "EY8x-fyQkbk_002885": {
    "video_start": 2885,
    "video_end": 2970,
    "anomaly_start": 4,
    "anomaly_end": 56,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 86,
    "subset": "val"
  },
  "EY8x-fyQkbk_003095": {
    "video_start": 3095,
    "video_end": 3245,
    "anomaly_start": 27,
    "anomaly_end": 138,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 151,
    "subset": "val"
  },
  "EY8x-fyQkbk_003863": {
    "video_start": 3863,
    "video_end": 3977,
    "anomaly_start": 26,
    "anomaly_end": 67,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 115,
    "subset": "val"
  },
  "EY8x-fyQkbk_004412": {
    "video_start": 4412,
    "video_end": 4478,
    "anomaly_start": 30,
    "anomaly_end": 60,
    "anomaly_class": "ego: turning",
    "num_frames": 67,
    "subset": "val"
  },
  "EY8x-fyQkbk_004480": {
    "video_start": 4480,
    "video_end": 4575,
    "anomaly_start": 28,
    "anomaly_end": 46,
    "anomaly_class": "other: lateral",
    "num_frames": 96,
    "subset": "val"
  },
  "EY8x-fyQkbk_004940": {
    "video_start": 4940,
    "video_end": 5027,
    "anomaly_start": 27,
    "anomaly_end": 75,
    "anomaly_class": "ego: turning",
    "num_frames": 88,
    "subset": "val"
  },
  "EY8x-fyQkbk_005029": {
    "video_start": 5029,
    "video_end": 5100,
    "anomaly_start": 42,
    "anomaly_end": 49,
    "anomaly_class": "ego: oncoming",
    "num_frames": 72,
    "subset": "val"
  },
  "Eq7_uD2yN5Y_000177": {
    "video_start": 177,
    "video_end": 265,
    "anomaly_start": 41,
    "anomaly_end": 58,
    "anomaly_class": "ego: oncoming",
    "num_frames": 89,
    "subset": "val"
  },
  "Eq7_uD2yN5Y_000267": {
    "video_start": 267,
    "video_end": 375,
    "anomaly_start": 45,
    "anomaly_end": 74,
    "anomaly_class": "ego: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "Eq7_uD2yN5Y_000377": {
    "video_start": 377,
    "video_end": 495,
    "anomaly_start": 53,
    "anomaly_end": 119,
    "anomaly_class": "ego: turning",
    "num_frames": 119,
    "subset": "val"
  },
  "Eq7_uD2yN5Y_000674": {
    "video_start": 674,
    "video_end": 772,
    "anomaly_start": 44,
    "anomaly_end": 77,
    "anomaly_class": "other: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "Eq7_uD2yN5Y_001004": {
    "video_start": 1004,
    "video_end": 1086,
    "anomaly_start": 27,
    "anomaly_end": 61,
    "anomaly_class": "ego: oncoming",
    "num_frames": 83,
    "subset": "val"
  },
  "Eq7_uD2yN5Y_001392": {
    "video_start": 1392,
    "video_end": 1519,
    "anomaly_start": 45,
    "anomaly_end": 75,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 128,
    "subset": "val"
  },
  "Eq7_uD2yN5Y_001521": {
    "video_start": 1521,
    "video_end": 1609,
    "anomaly_start": 46,
    "anomaly_end": 60,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "Eq7_uD2yN5Y_001731": {
    "video_start": 1731,
    "video_end": 1839,
    "anomaly_start": 36,
    "anomaly_end": 74,
    "anomaly_class": "ego: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "Eq7_uD2yN5Y_002416": {
    "video_start": 2416,
    "video_end": 2504,
    "anomaly_start": 39,
    "anomaly_end": 89,
    "anomaly_class": "ego: obstacle",
    "num_frames": 89,
    "subset": "val"
  },
  "Eq7_uD2yN5Y_002987": {
    "video_start": 2987,
    "video_end": 3098,
    "anomaly_start": 38,
    "anomaly_end": 89,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 112,
    "subset": "val"
  },
  "Eq7_uD2yN5Y_003100": {
    "video_start": 3100,
    "video_end": 3198,
    "anomaly_start": 40,
    "anomaly_end": 54,
    "anomaly_class": "other: oncoming",
    "num_frames": 99,
    "subset": "val"
  },
  "Eq7_uD2yN5Y_004139": {
    "video_start": 4139,
    "video_end": 4237,
    "anomaly_start": 56,
    "anomaly_end": 80,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 99,
    "subset": "val"
  },
  "Eq7_uD2yN5Y_004339": {
    "video_start": 4339,
    "video_end": 4417,
    "anomaly_start": 29,
    "anomaly_end": 44,
    "anomaly_class": "other: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "Eq7_uD2yN5Y_004509": {
    "video_start": 4509,
    "video_end": 4597,
    "anomaly_start": 40,
    "anomaly_end": 66,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "Eq7_uD2yN5Y_004599": {
    "video_start": 4599,
    "video_end": 4707,
    "anomaly_start": 42,
    "anomaly_end": 78,
    "anomaly_class": "other: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "Eq7_uD2yN5Y_004709": {
    "video_start": 4709,
    "video_end": 4810,
    "anomaly_start": 41,
    "anomaly_end": 88,
    "anomaly_class": "ego: lateral",
    "num_frames": 102,
    "subset": "val"
  },
  "Eq7_uD2yN5Y_005052": {
    "video_start": 5052,
    "video_end": 5191,
    "anomaly_start": 41,
    "anomaly_end": 75,
    "anomaly_class": "ego: pedestrian",
    "num_frames": 140,
    "subset": "val"
  },
  "F9-aR2Ocso4_001016": {
    "video_start": 1016,
    "video_end": 1100,
    "anomaly_start": 28,
    "anomaly_end": 55,
    "anomaly_class": "ego: lateral",
    "num_frames": 85,
    "subset": "val"
  },
  "FqJ4IfRLKmA_000441": {
    "video_start": 441,
    "video_end": 589,
    "anomaly_start": 71,
    "anomaly_end": 135,
    "anomaly_class": "ego: turning",
    "num_frames": 149,
    "subset": "val"
  },
  "FqJ4IfRLKmA_000951": {
    "video_start": 951,
    "video_end": 1039,
    "anomaly_start": 38,
    "anomaly_end": 63,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "FqJ4IfRLKmA_001141": {
    "video_start": 1141,
    "video_end": 1279,
    "anomaly_start": 80,
    "anomaly_end": 107,
    "anomaly_class": "other: start_stop_or_stationary",
    "num_frames": 139,
    "subset": "val"
  },
  "FqJ4IfRLKmA_001281": {
    "video_start": 1281,
    "video_end": 1429,
    "anomaly_start": 81,
    "anomaly_end": 114,
    "anomaly_class": "ego: oncoming",
    "num_frames": 149,
    "subset": "val"
  },
  "FqJ4IfRLKmA_004362": {
    "video_start": 4362,
    "video_end": 4480,
    "anomaly_start": 35,
    "anomaly_end": 119,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 119,
    "subset": "val"
  },
  "GDejMONyY4U_000423": {
    "video_start": 423,
    "video_end": 488,
    "anomaly_start": 30,
    "anomaly_end": 62,
    "anomaly_class": "ego: turning",
    "num_frames": 66,
    "subset": "val"
  },
  "GDejMONyY4U_003388": {
    "video_start": 3388,
    "video_end": 3495,
    "anomaly_start": 56,
    "anomaly_end": 94,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 108,
    "subset": "val"
  },
  "GDejMONyY4U_003618": {
    "video_start": 3618,
    "video_end": 3690,
    "anomaly_start": 26,
    "anomaly_end": 45,
    "anomaly_class": "ego: turning",
    "num_frames": 73,
    "subset": "val"
  },
  "GDejMONyY4U_004837": {
    "video_start": 4837,
    "video_end": 4901,
    "anomaly_start": 28,
    "anomaly_end": 54,
    "anomaly_class": "ego: turning",
    "num_frames": 65,
    "subset": "val"
  },
  "GDejMONyY4U_005327": {
    "video_start": 5327,
    "video_end": 5386,
    "anomaly_start": 26,
    "anomaly_end": 47,
    "anomaly_class": "other: turning",
    "num_frames": 60,
    "subset": "val"
  },
  "GJr8wKk8-YA_004465": {
    "video_start": 4465,
    "video_end": 4529,
    "anomaly_start": 21,
    "anomaly_end": 53,
    "anomaly_class": "other: turning",
    "num_frames": 65,
    "subset": "val"
  },
  "GLZENQ5lzGA_000021": {
    "video_start": 21,
    "video_end": 127,
    "anomaly_start": 46,
    "anomaly_end": 74,
    "anomaly_class": "ego: oncoming",
    "num_frames": 107,
    "subset": "val"
  },
  "GRDP68ucrBM_000061": {
    "video_start": 61,
    "video_end": 140,
    "anomaly_start": 28,
    "anomaly_end": 70,
    "anomaly_class": "ego: turning",
    "num_frames": 80,
    "subset": "val"
  },
  "GYWPqbrY8Z4_000809": {
    "video_start": 809,
    "video_end": 911,
    "anomaly_start": 54,
    "anomaly_end": 102,
    "anomaly_class": "ego: turning",
    "num_frames": 103,
    "subset": "val"
  },
  "GdyJqgxlSxs_000747": {
    "video_start": 747,
    "video_end": 808,
    "anomaly_start": 7,
    "anomaly_end": 44,
    "anomaly_class": "ego: turning",
    "num_frames": 62,
    "subset": "val"
  },
  "GdyJqgxlSxs_001451": {
    "video_start": 1451,
    "video_end": 1517,
    "anomaly_start": 28,
    "anomaly_end": 54,
    "anomaly_class": "ego: oncoming",
    "num_frames": 67,
    "subset": "val"
  },
  "GdyJqgxlSxs_001538": {
    "video_start": 1538,
    "video_end": 1606,
    "anomaly_start": 42,
    "anomaly_end": 58,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 69,
    "subset": "val"
  },
  "GdyJqgxlSxs_001912": {
    "video_start": 1912,
    "video_end": 2003,
    "anomaly_start": 19,
    "anomaly_end": 73,
    "anomaly_class": "ego: lateral",
    "num_frames": 92,
    "subset": "val"
  },
  "GdyJqgxlSxs_003144": {
    "video_start": 3144,
    "video_end": 3226,
    "anomaly_start": 35,
    "anomaly_end": 51,
    "anomaly_class": "ego: turning",
    "num_frames": 83,
    "subset": "val"
  },
  "GdyJqgxlSxs_005638": {
    "video_start": 5638,
    "video_end": 5726,
    "anomaly_start": 20,
    "anomaly_end": 66,
    "anomaly_class": "ego: lateral",
    "num_frames": 89,
    "subset": "val"
  },
  "GdyJqgxlSxs_006828": {
    "video_start": 6828,
    "video_end": 6902,
    "anomaly_start": 39,
    "anomaly_end": 69,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 75,
    "subset": "val"
  },
  "HNRS3w5zep8_000181": {
    "video_start": 181,
    "video_end": 278,
    "anomaly_start": 23,
    "anomaly_end": 70,
    "anomaly_class": "ego: lateral",
    "num_frames": 98,
    "subset": "val"
  },
  "HNRS3w5zep8_000299": {
    "video_start": 299,
    "video_end": 387,
    "anomaly_start": 1,
    "anomaly_end": 73,
    "anomaly_class": "ego: lateral",
    "num_frames": 89,
    "subset": "val"
  },
  "HNRS3w5zep8_001261": {
    "video_start": 1261,
    "video_end": 1345,
    "anomaly_start": 12,
    "anomaly_end": 47,
    "anomaly_class": "other: turning",
    "num_frames": 85,
    "subset": "val"
  },
  "HNRS3w5zep8_005441": {
    "video_start": 5441,
    "video_end": 5507,
    "anomaly_start": 19,
    "anomaly_end": 44,
    "anomaly_class": "ego: turning",
    "num_frames": 67,
    "subset": "val"
  },
  "HaC3LrJiTmQ_000307": {
    "video_start": 307,
    "video_end": 398,
    "anomaly_start": 49,
    "anomaly_end": 65,
    "anomaly_class": "other: turning",
    "num_frames": 92,
    "subset": "val"
  },
  "HaC3LrJiTmQ_001088": {
    "video_start": 1088,
    "video_end": 1208,
    "anomaly_start": 61,
    "anomaly_end": 90,
    "anomaly_class": "other: start_stop_or_stationary",
    "num_frames": 121,
    "subset": "val"
  },
  "HaC3LrJiTmQ_002545": {
    "video_start": 2545,
    "video_end": 2718,
    "anomaly_start": 15,
    "anomaly_end": 158,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 174,
    "subset": "val"
  },
  "HaC3LrJiTmQ_002935": {
    "video_start": 2935,
    "video_end": 3074,
    "anomaly_start": 39,
    "anomaly_end": 81,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 140,
    "subset": "val"
  },
  "HaC3LrJiTmQ_005242": {
    "video_start": 5242,
    "video_end": 5339,
    "anomaly_start": 45,
    "anomaly_end": 77,
    "anomaly_class": "ego: lateral",
    "num_frames": 98,
    "subset": "val"
  },
  "HaC3LrJiTmQ_005454": {
    "video_start": 5454,
    "video_end": 5502,
    "anomaly_start": 16,
    "anomaly_end": 49,
    "anomaly_class": "ego: unknown",
    "num_frames": 49,
    "subset": "val"
  },
  "Hd2IzHAfkCI_000301": {
    "video_start": 301,
    "video_end": 399,
    "anomaly_start": 49,
    "anomaly_end": 89,
    "anomaly_class": "ego: oncoming",
    "num_frames": 99,
    "subset": "val"
  },
  "Hd2IzHAfkCI_000719": {
    "video_start": 719,
    "video_end": 797,
    "anomaly_start": 35,
    "anomaly_end": 66,
    "anomaly_class": "ego: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "Hd2IzHAfkCI_001091": {
    "video_start": 1091,
    "video_end": 1199,
    "anomaly_start": 27,
    "anomaly_end": 103,
    "anomaly_class": "ego: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "Hd2IzHAfkCI_001392": {
    "video_start": 1392,
    "video_end": 1470,
    "anomaly_start": 38,
    "anomaly_end": 61,
    "anomaly_class": "other: oncoming",
    "num_frames": 79,
    "subset": "val"
  },
  "Hd2IzHAfkCI_001762": {
    "video_start": 1762,
    "video_end": 1870,
    "anomaly_start": 57,
    "anomaly_end": 87,
    "anomaly_class": "ego: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "Hd2IzHAfkCI_002211": {
    "video_start": 2211,
    "video_end": 2349,
    "anomaly_start": 1,
    "anomaly_end": 133,
    "anomaly_class": "ego: lateral",
    "num_frames": 139,
    "subset": "val"
  },
  "Hd2IzHAfkCI_002471": {
    "video_start": 2471,
    "video_end": 2579,
    "anomaly_start": 26,
    "anomaly_end": 71,
    "anomaly_class": "ego: lateral",
    "num_frames": 109,
    "subset": "val"
  },
  "Hd2IzHAfkCI_004081": {
    "video_start": 4081,
    "video_end": 4149,
    "anomaly_start": 22,
    "anomaly_end": 56,
    "anomaly_class": "ego: turning",
    "num_frames": 69,
    "subset": "val"
  },
  "Hd2IzHAfkCI_004231": {
    "video_start": 4231,
    "video_end": 4319,
    "anomaly_start": 19,
    "anomaly_end": 45,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "Hd2IzHAfkCI_004784": {
    "video_start": 4784,
    "video_end": 4882,
    "anomaly_start": 43,
    "anomaly_end": 79,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 99,
    "subset": "val"
  },
  "Hd2IzHAfkCI_005013": {
    "video_start": 5013,
    "video_end": 5111,
    "anomaly_start": 40,
    "anomaly_end": 91,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 99,
    "subset": "val"
  },
  "Hd2IzHAfkCI_005313": {
    "video_start": 5313,
    "video_end": 5391,
    "anomaly_start": 17,
    "anomaly_end": 47,
    "anomaly_class": "other: pedestrian",
    "num_frames": 79,
    "subset": "val"
  },
  "Hd2IzHAfkCI_005753": {
    "video_start": 5753,
    "video_end": 5869,
    "anomaly_start": 80,
    "anomaly_end": 99,
    "anomaly_class": "ego: start_stop_or_stationary",
    "num_frames": 117,
    "subset": "val"
  },
  "HeK6zbf8I9Y_000313": {
    "video_start": 313,
    "video_end": 411,
    "anomaly_start": 16,
    "anomaly_end": 38,
    "anomaly_class": "other: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "HeK6zbf8I9Y_004041": {
    "video_start": 4041,
    "video_end": 4103,
    "anomaly_start": 24,
    "anomaly_end": 46,
    "anomaly_class": "ego: turning",
    "num_frames": 63,
    "subset": "val"
  },
  "Hfs4AlBmg70_005511": {
    "video_start": 5511,
    "video_end": 5581,
    "anomaly_start": 15,
    "anomaly_end": 48,
    "anomaly_class": "other: lateral",
    "num_frames": 71,
    "subset": "val"
  },
  "Hfs4AlBmg70_006168": {
    "video_start": 6168,
    "video_end": 6220,
    "anomaly_start": 26,
    "anomaly_end": 41,
    "anomaly_class": "ego: turning",
    "num_frames": 53,
    "subset": "val"
  },
  "Hkui6PJboFg_000498": {
    "video_start": 498,
    "video_end": 607,
    "anomaly_start": 26,
    "anomaly_end": 33,
    "anomaly_class": "ego: oncoming",
    "num_frames": 110,
    "subset": "val"
  },
  "Hkui6PJboFg_001314": {
    "video_start": 1314,
    "video_end": 1384,
    "anomaly_start": 13,
    "anomaly_end": 50,
    "anomaly_class": "ego: turning",
    "num_frames": 71,
    "subset": "val"
  },
  "Hkui6PJboFg_001752": {
    "video_start": 1752,
    "video_end": 1908,
    "anomaly_start": 57,
    "anomaly_end": 98,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 157,
    "subset": "val"
  },
  "Hkui6PJboFg_002014": {
    "video_start": 2014,
    "video_end": 2097,
    "anomaly_start": 32,
    "anomaly_end": 60,
    "anomaly_class": "other: turning",
    "num_frames": 84,
    "subset": "val"
  },
  "Hkui6PJboFg_002181": {
    "video_start": 2181,
    "video_end": 2257,
    "anomaly_start": 17,
    "anomaly_end": 39,
    "anomaly_class": "ego: lateral",
    "num_frames": 77,
    "subset": "val"
  },
  "Hkui6PJboFg_002486": {
    "video_start": 2486,
    "video_end": 2576,
    "anomaly_start": 39,
    "anomaly_end": 67,
    "anomaly_class": "ego: turning",
    "num_frames": 91,
    "subset": "val"
  },
  "Hkui6PJboFg_003202": {
    "video_start": 3202,
    "video_end": 3306,
    "anomaly_start": 52,
    "anomaly_end": 93,
    "anomaly_class": "ego: unknown",
    "num_frames": 105,
    "subset": "val"
  },
  "Hkui6PJboFg_004478": {
    "video_start": 4478,
    "video_end": 4553,
    "anomaly_start": 35,
    "anomaly_end": 57,
    "anomaly_class": "ego: obstacle",
    "num_frames": 76,
    "subset": "val"
  },
  "Hkui6PJboFg_004788": {
    "video_start": 4788,
    "video_end": 4846,
    "anomaly_start": 15,
    "anomaly_end": 43,
    "anomaly_class": "ego: lateral",
    "num_frames": 59,
    "subset": "val"
  },
  "Hkui6PJboFg_004848": {
    "video_start": 4848,
    "video_end": 4895,
    "anomaly_start": 10,
    "anomaly_end": 43,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 48,
    "subset": "val"
  },
  "Hkui6PJboFg_005596": {
    "video_start": 5596,
    "video_end": 5666,
    "anomaly_start": 22,
    "anomaly_end": 56,
    "anomaly_class": "other: unknown",
    "num_frames": 71,
    "subset": "val"
  },
  "Hkui6PJboFg_005957": {
    "video_start": 5957,
    "video_end": 6030,
    "anomaly_start": 24,
    "anomaly_end": 62,
    "anomaly_class": "ego: turning",
    "num_frames": 74,
    "subset": "val"
  },
  "Hx8FMhmdOQU_000173": {
    "video_start": 173,
    "video_end": 261,
    "anomaly_start": 31,
    "anomaly_end": 74,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "Hx8FMhmdOQU_000483": {
    "video_start": 483,
    "video_end": 603,
    "anomaly_start": 75,
    "anomaly_end": 121,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 121,
    "subset": "val"
  },
  "Hx8FMhmdOQU_001107": {
    "video_start": 1107,
    "video_end": 1206,
    "anomaly_start": 35,
    "anomaly_end": 67,
    "anomaly_class": "ego: turning",
    "num_frames": 100,
    "subset": "val"
  },
  "Hx8FMhmdOQU_001484": {
    "video_start": 1484,
    "video_end": 1607,
    "anomaly_start": 27,
    "anomaly_end": 120,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 124,
    "subset": "val"
  },
  "Hx8FMhmdOQU_001609": {
    "video_start": 1609,
    "video_end": 1787,
    "anomaly_start": 43,
    "anomaly_end": 168,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 179,
    "subset": "val"
  },
  "Hx8FMhmdOQU_002029": {
    "video_start": 2029,
    "video_end": 2152,
    "anomaly_start": 22,
    "anomaly_end": 60,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 124,
    "subset": "val"
  },
  "Hx8FMhmdOQU_003601": {
    "video_start": 3601,
    "video_end": 3689,
    "anomaly_start": 21,
    "anomaly_end": 67,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "Hx8FMhmdOQU_003781": {
    "video_start": 3781,
    "video_end": 3850,
    "anomaly_start": 35,
    "anomaly_end": 69,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 70,
    "subset": "val"
  },
  "Hx8FMhmdOQU_004252": {
    "video_start": 4252,
    "video_end": 4346,
    "anomaly_start": 49,
    "anomaly_end": 57,
    "anomaly_class": "ego: oncoming",
    "num_frames": 95,
    "subset": "val"
  },
  "Hx8FMhmdOQU_005600": {
    "video_start": 5600,
    "video_end": 5698,
    "anomaly_start": 39,
    "anomaly_end": 70,
    "anomaly_class": "ego: pedestrian",
    "num_frames": 99,
    "subset": "val"
  },
  "HzVbo46kkBA_000337": {
    "video_start": 337,
    "video_end": 445,
    "anomaly_start": 66,
    "anomaly_end": 95,
    "anomaly_class": "ego: obstacle",
    "num_frames": 109,
    "subset": "val"
  },
  "HzVbo46kkBA_000817": {
    "video_start": 817,
    "video_end": 916,
    "anomaly_start": 28,
    "anomaly_end": 75,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 100,
    "subset": "val"
  },
  "HzVbo46kkBA_000918": {
    "video_start": 918,
    "video_end": 1026,
    "anomaly_start": 41,
    "anomaly_end": 61,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 109,
    "subset": "val"
  },
  "HzVbo46kkBA_001622": {
    "video_start": 1622,
    "video_end": 1685,
    "anomaly_start": 13,
    "anomaly_end": 30,
    "anomaly_class": "other: turning",
    "num_frames": 64,
    "subset": "val"
  },
  "HzVbo46kkBA_001687": {
    "video_start": 1687,
    "video_end": 1748,
    "anomaly_start": 3,
    "anomaly_end": 32,
    "anomaly_class": "ego: lateral",
    "num_frames": 62,
    "subset": "val"
  },
  "HzVbo46kkBA_001750": {
    "video_start": 1750,
    "video_end": 1859,
    "anomaly_start": 27,
    "anomaly_end": 109,
    "anomaly_class": "other: lateral",
    "num_frames": 110,
    "subset": "val"
  },
  "HzVbo46kkBA_001987": {
    "video_start": 1987,
    "video_end": 2075,
    "anomaly_start": 21,
    "anomaly_end": 44,
    "anomaly_class": "other: lateral",
    "num_frames": 89,
    "subset": "val"
  },
  "HzVbo46kkBA_002328": {
    "video_start": 2328,
    "video_end": 2475,
    "anomaly_start": 74,
    "anomaly_end": 133,
    "anomaly_class": "ego: turning",
    "num_frames": 148,
    "subset": "val"
  },
  "HzVbo46kkBA_003111": {
    "video_start": 3111,
    "video_end": 3238,
    "anomaly_start": 34,
    "anomaly_end": 76,
    "anomaly_class": "ego: turning",
    "num_frames": 128,
    "subset": "val"
  },
  "HzVbo46kkBA_003934": {
    "video_start": 3934,
    "video_end": 4042,
    "anomaly_start": 33,
    "anomaly_end": 103,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 109,
    "subset": "val"
  },
  "HzVbo46kkBA_004124": {
    "video_start": 4124,
    "video_end": 4223,
    "anomaly_start": 40,
    "anomaly_end": 75,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 100,
    "subset": "val"
  },
  "HzVbo46kkBA_004323": {
    "video_start": 4323,
    "video_end": 4441,
    "anomaly_start": 41,
    "anomaly_end": 68,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 119,
    "subset": "val"
  },
  "HzVbo46kkBA_004563": {
    "video_start": 4563,
    "video_end": 4733,
    "anomaly_start": 19,
    "anomaly_end": 154,
    "anomaly_class": "other: turning",
    "num_frames": 171,
    "subset": "val"
  },
  "HzVbo46kkBA_005648": {
    "video_start": 5648,
    "video_end": 5825,
    "anomaly_start": 136,
    "anomaly_end": 175,
    "anomaly_class": "ego: turning",
    "num_frames": 178,
    "subset": "val"
  },
  "IFhWIS-Gwps_000227": {
    "video_start": 227,
    "video_end": 367,
    "anomaly_start": 84,
    "anomaly_end": 122,
    "anomaly_class": "ego: oncoming",
    "num_frames": 141,
    "subset": "val"
  },
  "IIuaTrT3pp0_000498": {
    "video_start": 498,
    "video_end": 573,
    "anomaly_start": 17,
    "anomaly_end": 67,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 76,
    "subset": "val"
  },
  "IOoVklIZLMw_000945": {
    "video_start": 945,
    "video_end": 1061,
    "anomaly_start": 50,
    "anomaly_end": 86,
    "anomaly_class": "ego: oncoming",
    "num_frames": 117,
    "subset": "val"
  },
  "IOoVklIZLMw_003435": {
    "video_start": 3435,
    "video_end": 3543,
    "anomaly_start": 37,
    "anomaly_end": 53,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 109,
    "subset": "val"
  },
  "IQwS7BEjUxM_000147": {
    "video_start": 147,
    "video_end": 219,
    "anomaly_start": 10,
    "anomaly_end": 40,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 73,
    "subset": "val"
  },
  "IQwS7BEjUxM_000423": {
    "video_start": 423,
    "video_end": 513,
    "anomaly_start": 31,
    "anomaly_end": 38,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 91,
    "subset": "val"
  },
  "IQwS7BEjUxM_001382": {
    "video_start": 1382,
    "video_end": 1489,
    "anomaly_start": 58,
    "anomaly_end": 95,
    "anomaly_class": "ego: lateral",
    "num_frames": 108,
    "subset": "val"
  },
  "IQwS7BEjUxM_001491": {
    "video_start": 1491,
    "video_end": 1638,
    "anomaly_start": 59,
    "anomaly_end": 126,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 148,
    "subset": "val"
  },
  "IQwS7BEjUxM_001783": {
    "video_start": 1783,
    "video_end": 1876,
    "anomaly_start": 41,
    "anomaly_end": 88,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 94,
    "subset": "val"
  },
  "IQwS7BEjUxM_002096": {
    "video_start": 2096,
    "video_end": 2237,
    "anomaly_start": 34,
    "anomaly_end": 95,
    "anomaly_class": "other: lateral",
    "num_frames": 142,
    "subset": "val"
  },
  "IQwS7BEjUxM_002239": {
    "video_start": 2239,
    "video_end": 2377,
    "anomaly_start": 57,
    "anomaly_end": 117,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 139,
    "subset": "val"
  },
  "IQwS7BEjUxM_002546": {
    "video_start": 2546,
    "video_end": 2664,
    "anomaly_start": 62,
    "anomaly_end": 92,
    "anomaly_class": "ego: turning",
    "num_frames": 119,
    "subset": "val"
  },
  "IQwS7BEjUxM_003370": {
    "video_start": 3370,
    "video_end": 3468,
    "anomaly_start": 21,
    "anomaly_end": 44,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 99,
    "subset": "val"
  },
  "Izf9l3VWQuo_000815": {
    "video_start": 815,
    "video_end": 923,
    "anomaly_start": 18,
    "anomaly_end": 83,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 109,
    "subset": "val"
  },
  "Izf9l3VWQuo_000925": {
    "video_start": 925,
    "video_end": 984,
    "anomaly_start": 38,
    "anomaly_end": 60,
    "anomaly_class": "ego: lateral",
    "num_frames": 60,
    "subset": "val"
  },
  "Izf9l3VWQuo_001419": {
    "video_start": 1419,
    "video_end": 1568,
    "anomaly_start": 40,
    "anomaly_end": 116,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 150,
    "subset": "val"
  },
  "Izf9l3VWQuo_001570": {
    "video_start": 1570,
    "video_end": 1649,
    "anomaly_start": 22,
    "anomaly_end": 52,
    "anomaly_class": "other: oncoming",
    "num_frames": 80,
    "subset": "val"
  },
  "Izf9l3VWQuo_002029": {
    "video_start": 2029,
    "video_end": 2150,
    "anomaly_start": 50,
    "anomaly_end": 88,
    "anomaly_class": "other: oncoming",
    "num_frames": 122,
    "subset": "val"
  },
  "Izf9l3VWQuo_003613": {
    "video_start": 3613,
    "video_end": 3766,
    "anomaly_start": 73,
    "anomaly_end": 92,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 154,
    "subset": "val"
  },
  "Izf9l3VWQuo_004280": {
    "video_start": 4280,
    "video_end": 4420,
    "anomaly_start": 70,
    "anomaly_end": 125,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 141,
    "subset": "val"
  },
  "Izf9l3VWQuo_005326": {
    "video_start": 5326,
    "video_end": 5434,
    "anomaly_start": 31,
    "anomaly_end": 50,
    "anomaly_class": "other: oncoming",
    "num_frames": 109,
    "subset": "val"
  },
  "Izf9l3VWQuo_005523": {
    "video_start": 5523,
    "video_end": 5632,
    "anomaly_start": 17,
    "anomaly_end": 73,
    "anomaly_class": "other: turning",
    "num_frames": 110,
    "subset": "val"
  },
  "Izf9l3VWQuo_005634": {
    "video_start": 5634,
    "video_end": 5732,
    "anomaly_start": 39,
    "anomaly_end": 60,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 99,
    "subset": "val"
  },
  "J-j5eIHg8I0_000194": {
    "video_start": 194,
    "video_end": 251,
    "anomaly_start": 54,
    "anomaly_end": 58,
    "anomaly_class": "ego: turning",
    "num_frames": 58,
    "subset": "val"
  },
  "J-j5eIHg8I0_000723": {
    "video_start": 723,
    "video_end": 863,
    "anomaly_start": 67,
    "anomaly_end": 82,
    "anomaly_class": "other: turning",
    "num_frames": 141,
    "subset": "val"
  },
  "J-j5eIHg8I0_000865": {
    "video_start": 865,
    "video_end": 992,
    "anomaly_start": 38,
    "anomaly_end": 86,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 128,
    "subset": "val"
  },
  "J-j5eIHg8I0_000994": {
    "video_start": 994,
    "video_end": 1132,
    "anomaly_start": 53,
    "anomaly_end": 104,
    "anomaly_class": "ego: turning",
    "num_frames": 139,
    "subset": "val"
  },
  "J-j5eIHg8I0_001134": {
    "video_start": 1134,
    "video_end": 1284,
    "anomaly_start": 24,
    "anomaly_end": 56,
    "anomaly_class": "other: lateral",
    "num_frames": 151,
    "subset": "val"
  },
  "J-j5eIHg8I0_001286": {
    "video_start": 1286,
    "video_end": 1442,
    "anomaly_start": 87,
    "anomaly_end": 129,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 157,
    "subset": "val"
  },
  "J-j5eIHg8I0_001514": {
    "video_start": 1514,
    "video_end": 1652,
    "anomaly_start": 60,
    "anomaly_end": 90,
    "anomaly_class": "other: turning",
    "num_frames": 139,
    "subset": "val"
  },
  "J-j5eIHg8I0_001654": {
    "video_start": 1654,
    "video_end": 1694,
    "anomaly_start": 21,
    "anomaly_end": 41,
    "anomaly_class": "other: turning",
    "num_frames": 41,
    "subset": "val"
  },
  "J-j5eIHg8I0_001696": {
    "video_start": 1696,
    "video_end": 1814,
    "anomaly_start": 26,
    "anomaly_end": 86,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 119,
    "subset": "val"
  },
  "J-j5eIHg8I0_001816": {
    "video_start": 1816,
    "video_end": 1915,
    "anomaly_start": 31,
    "anomaly_end": 61,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 100,
    "subset": "val"
  },
  "J-j5eIHg8I0_002477": {
    "video_start": 2477,
    "video_end": 2586,
    "anomaly_start": 37,
    "anomaly_end": 79,
    "anomaly_class": "ego: turning",
    "num_frames": 110,
    "subset": "val"
  },
  "J-j5eIHg8I0_004416": {
    "video_start": 4416,
    "video_end": 4490,
    "anomaly_start": 24,
    "anomaly_end": 50,
    "anomaly_class": "other: turning",
    "num_frames": 75,
    "subset": "val"
  },
  "J-vWxzY9H9E_000271": {
    "video_start": 271,
    "video_end": 359,
    "anomaly_start": 38,
    "anomaly_end": 71,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "JHfjuFrnpjA_000441": {
    "video_start": 441,
    "video_end": 549,
    "anomaly_start": 49,
    "anomaly_end": 68,
    "anomaly_class": "other: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "JHfjuFrnpjA_000551": {
    "video_start": 551,
    "video_end": 629,
    "anomaly_start": 10,
    "anomaly_end": 37,
    "anomaly_class": "other: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "JHfjuFrnpjA_000721": {
    "video_start": 721,
    "video_end": 860,
    "anomaly_start": 73,
    "anomaly_end": 118,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 140,
    "subset": "val"
  },
  "JHfjuFrnpjA_001044": {
    "video_start": 1044,
    "video_end": 1162,
    "anomaly_start": 33,
    "anomaly_end": 70,
    "anomaly_class": "other: turning",
    "num_frames": 119,
    "subset": "val"
  },
  "JHfjuFrnpjA_001164": {
    "video_start": 1164,
    "video_end": 1278,
    "anomaly_start": 12,
    "anomaly_end": 98,
    "anomaly_class": "other: lateral",
    "num_frames": 115,
    "subset": "val"
  },
  "JHfjuFrnpjA_002290": {
    "video_start": 2290,
    "video_end": 2402,
    "anomaly_start": 37,
    "anomaly_end": 68,
    "anomaly_class": "other: lateral",
    "num_frames": 113,
    "subset": "val"
  },
  "JHfjuFrnpjA_005229": {
    "video_start": 5229,
    "video_end": 5317,
    "anomaly_start": 40,
    "anomaly_end": 61,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "JHfjuFrnpjA_005930": {
    "video_start": 5930,
    "video_end": 6017,
    "anomaly_start": 48,
    "anomaly_end": 70,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 88,
    "subset": "val"
  },
  "JV0D-YkWHD8_000689": {
    "video_start": 689,
    "video_end": 755,
    "anomaly_start": 0,
    "anomaly_end": 42,
    "anomaly_class": "ego: lateral",
    "num_frames": 67,
    "subset": "val"
  },
  "JV0D-YkWHD8_000851": {
    "video_start": 851,
    "video_end": 954,
    "anomaly_start": 43,
    "anomaly_end": 61,
    "anomaly_class": "ego: turning",
    "num_frames": 104,
    "subset": "val"
  },
  "JV0D-YkWHD8_001349": {
    "video_start": 1349,
    "video_end": 1452,
    "anomaly_start": 22,
    "anomaly_end": 63,
    "anomaly_class": "ego: lateral",
    "num_frames": 104,
    "subset": "val"
  },
  "JV0D-YkWHD8_001454": {
    "video_start": 1454,
    "video_end": 1529,
    "anomaly_start": 21,
    "anomaly_end": 45,
    "anomaly_class": "other: turning",
    "num_frames": 76,
    "subset": "val"
  },
  "JV0D-YkWHD8_001531": {
    "video_start": 1531,
    "video_end": 1597,
    "anomaly_start": 17,
    "anomaly_end": 42,
    "anomaly_class": "ego: turning",
    "num_frames": 67,
    "subset": "val"
  },
  "JV0D-YkWHD8_001686": {
    "video_start": 1686,
    "video_end": 1783,
    "anomaly_start": 43,
    "anomaly_end": 86,
    "anomaly_class": "ego: pedestrian",
    "num_frames": 98,
    "subset": "val"
  },
  "JV0D-YkWHD8_002421": {
    "video_start": 2421,
    "video_end": 2514,
    "anomaly_start": 34,
    "anomaly_end": 76,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 94,
    "subset": "val"
  },
  "JV0D-YkWHD8_003492": {
    "video_start": 3492,
    "video_end": 3558,
    "anomaly_start": 23,
    "anomaly_end": 38,
    "anomaly_class": "ego: lateral",
    "num_frames": 67,
    "subset": "val"
  },
  "JV0D-YkWHD8_003654": {
    "video_start": 3654,
    "video_end": 3742,
    "anomaly_start": 25,
    "anomaly_end": 78,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 89,
    "subset": "val"
  },
  "JV0D-YkWHD8_003939": {
    "video_start": 3939,
    "video_end": 4031,
    "anomaly_start": 34,
    "anomaly_end": 86,
    "anomaly_class": "other: lateral",
    "num_frames": 93,
    "subset": "val"
  },
  "J_ggxJD3Uy8_004713": {
    "video_start": 4713,
    "video_end": 4782,
    "anomaly_start": 38,
    "anomaly_end": 66,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 70,
    "subset": "val"
  },
  "JkYzYrJpSoQ_001255": {
    "video_start": 1255,
    "video_end": 1353,
    "anomaly_start": 58,
    "anomaly_end": 70,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 99,
    "subset": "val"
  },
  "JkYzYrJpSoQ_001556": {
    "video_start": 1556,
    "video_end": 1644,
    "anomaly_start": 28,
    "anomaly_end": 66,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "JkYzYrJpSoQ_002520": {
    "video_start": 2520,
    "video_end": 2608,
    "anomaly_start": 43,
    "anomaly_end": 62,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "JkYzYrJpSoQ_004208": {
    "video_start": 4208,
    "video_end": 4296,
    "anomaly_start": 42,
    "anomaly_end": 68,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "JkYzYrJpSoQ_004298": {
    "video_start": 4298,
    "video_end": 4396,
    "anomaly_start": 46,
    "anomaly_end": 70,
    "anomaly_class": "ego: lateral",
    "num_frames": 99,
    "subset": "val"
  },
  "JkYzYrJpSoQ_005468": {
    "video_start": 5468,
    "video_end": 5576,
    "anomaly_start": 64,
    "anomaly_end": 88,
    "anomaly_class": "other: unknown",
    "num_frames": 109,
    "subset": "val"
  },
  "JkYzYrJpSoQ_005578": {
    "video_start": 5578,
    "video_end": 5676,
    "anomaly_start": 39,
    "anomaly_end": 71,
    "anomaly_class": "ego: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "Jt6_hinpKpk_001303": {
    "video_start": 1303,
    "video_end": 1346,
    "anomaly_start": 9,
    "anomaly_end": 43,
    "anomaly_class": "other: turning",
    "num_frames": 44,
    "subset": "val"
  },
  "K1r3m5OrmB4_001155": {
    "video_start": 1155,
    "video_end": 1233,
    "anomaly_start": 19,
    "anomaly_end": 54,
    "anomaly_class": "other: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "K1r3m5OrmB4_001770": {
    "video_start": 1770,
    "video_end": 1918,
    "anomaly_start": 29,
    "anomaly_end": 76,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 149,
    "subset": "val"
  },
  "K1r3m5OrmB4_002157": {
    "video_start": 2157,
    "video_end": 2214,
    "anomaly_start": 22,
    "anomaly_end": 37,
    "anomaly_class": "other: lateral",
    "num_frames": 58,
    "subset": "val"
  },
  "K1r3m5OrmB4_002380": {
    "video_start": 2380,
    "video_end": 2472,
    "anomaly_start": 17,
    "anomaly_end": 75,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 93,
    "subset": "val"
  },
  "K1r3m5OrmB4_003389": {
    "video_start": 3389,
    "video_end": 3477,
    "anomaly_start": 20,
    "anomaly_end": 49,
    "anomaly_class": "ego: lateral",
    "num_frames": 89,
    "subset": "val"
  },
  "K1r3m5OrmB4_003479": {
    "video_start": 3479,
    "video_end": 3602,
    "anomaly_start": 8,
    "anomaly_end": 49,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 124,
    "subset": "val"
  },
  "K1r3m5OrmB4_003721": {
    "video_start": 3721,
    "video_end": 3814,
    "anomaly_start": 32,
    "anomaly_end": 75,
    "anomaly_class": "ego: lateral",
    "num_frames": 94,
    "subset": "val"
  },
  "K1r3m5OrmB4_004010": {
    "video_start": 4010,
    "video_end": 4097,
    "anomaly_start": 21,
    "anomaly_end": 69,
    "anomaly_class": "ego: turning",
    "num_frames": 88,
    "subset": "val"
  },
  "K1r3m5OrmB4_005770": {
    "video_start": 5770,
    "video_end": 5864,
    "anomaly_start": 54,
    "anomaly_end": 79,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 95,
    "subset": "val"
  },
  "K1r3m5OrmB4_005928": {
    "video_start": 5928,
    "video_end": 6003,
    "anomaly_start": 17,
    "anomaly_end": 51,
    "anomaly_class": "ego: lateral",
    "num_frames": 76,
    "subset": "val"
  },
  "K1r3m5OrmB4_006472": {
    "video_start": 6472,
    "video_end": 6546,
    "anomaly_start": 38,
    "anomaly_end": 75,
    "anomaly_class": "ego: turning",
    "num_frames": 75,
    "subset": "val"
  },
  "KOm5jw8vGrE_001190": {
    "video_start": 1190,
    "video_end": 1246,
    "anomaly_start": 36,
    "anomaly_end": 51,
    "anomaly_class": "ego: turning",
    "num_frames": 57,
    "subset": "val"
  },
  "L334aqEJxys_001996": {
    "video_start": 1996,
    "video_end": 2134,
    "anomaly_start": 47,
    "anomaly_end": 108,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 139,
    "subset": "val"
  },
  "L334aqEJxys_003180": {
    "video_start": 3180,
    "video_end": 3278,
    "anomaly_start": 36,
    "anomaly_end": 64,
    "anomaly_class": "ego: lateral",
    "num_frames": 99,
    "subset": "val"
  },
  "L334aqEJxys_004170": {
    "video_start": 4170,
    "video_end": 4295,
    "anomaly_start": 64,
    "anomaly_end": 96,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 126,
    "subset": "val"
  },
  "L334aqEJxys_004297": {
    "video_start": 4297,
    "video_end": 4385,
    "anomaly_start": 41,
    "anomaly_end": 74,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 89,
    "subset": "val"
  },
  "L334aqEJxys_004809": {
    "video_start": 4809,
    "video_end": 4992,
    "anomaly_start": 113,
    "anomaly_end": 184,
    "anomaly_class": "ego: pedestrian",
    "num_frames": 184,
    "subset": "val"
  },
  "LbKd5Ho0wDM_000635": {
    "video_start": 635,
    "video_end": 721,
    "anomaly_start": 39,
    "anomaly_end": 72,
    "anomaly_class": "other: lateral",
    "num_frames": 87,
    "subset": "val"
  },
  "LfKfK4I5RPE_000321": {
    "video_start": 321,
    "video_end": 419,
    "anomaly_start": 49,
    "anomaly_end": 70,
    "anomaly_class": "ego: oncoming",
    "num_frames": 99,
    "subset": "val"
  },
  "LfKfK4I5RPE_000722": {
    "video_start": 722,
    "video_end": 820,
    "anomaly_start": 56,
    "anomaly_end": 76,
    "anomaly_class": "ego: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "LfKfK4I5RPE_001441": {
    "video_start": 1441,
    "video_end": 1560,
    "anomaly_start": 16,
    "anomaly_end": 101,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 120,
    "subset": "val"
  },
  "LfKfK4I5RPE_001779": {
    "video_start": 1779,
    "video_end": 1887,
    "anomaly_start": 54,
    "anomaly_end": 90,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 109,
    "subset": "val"
  },
  "LfKfK4I5RPE_002296": {
    "video_start": 2296,
    "video_end": 2394,
    "anomaly_start": 45,
    "anomaly_end": 74,
    "anomaly_class": "ego: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "LfKfK4I5RPE_002937": {
    "video_start": 2937,
    "video_end": 2989,
    "anomaly_start": 36,
    "anomaly_end": 53,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 53,
    "subset": "val"
  },
  "LfKfK4I5RPE_003915": {
    "video_start": 3915,
    "video_end": 4003,
    "anomaly_start": 43,
    "anomaly_end": 66,
    "anomaly_class": "other: oncoming",
    "num_frames": 89,
    "subset": "val"
  },
  "LfKfK4I5RPE_004329": {
    "video_start": 4329,
    "video_end": 4407,
    "anomaly_start": 28,
    "anomaly_end": 50,
    "anomaly_class": "ego: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "LfKfK4I5RPE_005725": {
    "video_start": 5725,
    "video_end": 5901,
    "anomaly_start": 51,
    "anomaly_end": 153,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 177,
    "subset": "val"
  },
  "Nc9SDBksPPA_000075": {
    "video_start": 75,
    "video_end": 169,
    "anomaly_start": 37,
    "anomaly_end": 55,
    "anomaly_class": "other: turning",
    "num_frames": 95,
    "subset": "val"
  },
  "Nc9SDBksPPA_000531": {
    "video_start": 531,
    "video_end": 619,
    "anomaly_start": 26,
    "anomaly_end": 58,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "Nc9SDBksPPA_000740": {
    "video_start": 740,
    "video_end": 838,
    "anomaly_start": 43,
    "anomaly_end": 67,
    "anomaly_class": "other: lateral",
    "num_frames": 99,
    "subset": "val"
  },
  "Nc9SDBksPPA_000961": {
    "video_start": 961,
    "video_end": 1137,
    "anomaly_start": 40,
    "anomaly_end": 155,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 177,
    "subset": "val"
  },
  "Nc9SDBksPPA_001462": {
    "video_start": 1462,
    "video_end": 1571,
    "anomaly_start": 30,
    "anomaly_end": 88,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 110,
    "subset": "val"
  },
  "Nc9SDBksPPA_001801": {
    "video_start": 1801,
    "video_end": 1860,
    "anomaly_start": 38,
    "anomaly_end": 60,
    "anomaly_class": "other: lateral",
    "num_frames": 60,
    "subset": "val"
  },
  "Nc9SDBksPPA_002070": {
    "video_start": 2070,
    "video_end": 2185,
    "anomaly_start": 43,
    "anomaly_end": 92,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 116,
    "subset": "val"
  },
  "Nc9SDBksPPA_003515": {
    "video_start": 3515,
    "video_end": 3623,
    "anomaly_start": 44,
    "anomaly_end": 94,
    "anomaly_class": "ego: lateral",
    "num_frames": 109,
    "subset": "val"
  },
  "Nc9SDBksPPA_003625": {
    "video_start": 3625,
    "video_end": 3733,
    "anomaly_start": 43,
    "anomaly_end": 63,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 109,
    "subset": "val"
  },
  "Nc9SDBksPPA_003925": {
    "video_start": 3925,
    "video_end": 4003,
    "anomaly_start": 24,
    "anomaly_end": 50,
    "anomaly_class": "other: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "Nc9SDBksPPA_004594": {
    "video_start": 4594,
    "video_end": 4712,
    "anomaly_start": 68,
    "anomaly_end": 82,
    "anomaly_class": "ego: turning",
    "num_frames": 119,
    "subset": "val"
  },
  "O9uvBFovKj8_000893": {
    "video_start": 893,
    "video_end": 981,
    "anomaly_start": 23,
    "anomaly_end": 52,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "O9uvBFovKj8_001367": {
    "video_start": 1367,
    "video_end": 1465,
    "anomaly_start": 54,
    "anomaly_end": 75,
    "anomaly_class": "ego: oncoming",
    "num_frames": 99,
    "subset": "val"
  },
  "O9uvBFovKj8_001577": {
    "video_start": 1577,
    "video_end": 1676,
    "anomaly_start": 31,
    "anomaly_end": 79,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 100,
    "subset": "val"
  },
  "O9uvBFovKj8_001678": {
    "video_start": 1678,
    "video_end": 1799,
    "anomaly_start": 53,
    "anomaly_end": 106,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 122,
    "subset": "val"
  },
  "O9uvBFovKj8_002029": {
    "video_start": 2029,
    "video_end": 2107,
    "anomaly_start": 38,
    "anomaly_end": 58,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 79,
    "subset": "val"
  },
  "O9uvBFovKj8_003798": {
    "video_start": 3798,
    "video_end": 3886,
    "anomaly_start": 31,
    "anomaly_end": 59,
    "anomaly_class": "other: pedestrian",
    "num_frames": 89,
    "subset": "val"
  },
  "O9uvBFovKj8_004388": {
    "video_start": 4388,
    "video_end": 4515,
    "anomaly_start": 42,
    "anomaly_end": 76,
    "anomaly_class": "ego: turning",
    "num_frames": 128,
    "subset": "val"
  },
  "O9uvBFovKj8_004517": {
    "video_start": 4517,
    "video_end": 4630,
    "anomaly_start": 43,
    "anomaly_end": 92,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 114,
    "subset": "val"
  },
  "O9uvBFovKj8_004782": {
    "video_start": 4782,
    "video_end": 4880,
    "anomaly_start": 23,
    "anomaly_end": 67,
    "anomaly_class": "ego: lateral",
    "num_frames": 99,
    "subset": "val"
  },
  "O9uvBFovKj8_005244": {
    "video_start": 5244,
    "video_end": 5322,
    "anomaly_start": 19,
    "anomaly_end": 34,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 79,
    "subset": "val"
  },
  "O9uvBFovKj8_005324": {
    "video_start": 5324,
    "video_end": 5432,
    "anomaly_start": 41,
    "anomaly_end": 79,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 109,
    "subset": "val"
  },
  "O9uvBFovKj8_005434": {
    "video_start": 5434,
    "video_end": 5573,
    "anomaly_start": 24,
    "anomaly_end": 53,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 140,
    "subset": "val"
  },
  "OPj7lUQOre8_002571": {
    "video_start": 2571,
    "video_end": 2679,
    "anomaly_start": 45,
    "anomaly_end": 99,
    "anomaly_class": "other: lateral",
    "num_frames": 109,
    "subset": "val"
  },
  "OPj7lUQOre8_002681": {
    "video_start": 2681,
    "video_end": 2819,
    "anomaly_start": 56,
    "anomaly_end": 83,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 139,
    "subset": "val"
  },
  "OPj7lUQOre8_004526": {
    "video_start": 4526,
    "video_end": 4624,
    "anomaly_start": 39,
    "anomaly_end": 67,
    "anomaly_class": "ego: lateral",
    "num_frames": 99,
    "subset": "val"
  },
  "OPj7lUQOre8_005177": {
    "video_start": 5177,
    "video_end": 5285,
    "anomaly_start": 36,
    "anomaly_end": 55,
    "anomaly_class": "other: obstacle",
    "num_frames": 109,
    "subset": "val"
  },
  "OPj7lUQOre8_005647": {
    "video_start": 5647,
    "video_end": 5795,
    "anomaly_start": 61,
    "anomaly_end": 114,
    "anomaly_class": "ego: lateral",
    "num_frames": 149,
    "subset": "val"
  },
  "OWtbKblBOKI_001000": {
    "video_start": 1000,
    "video_end": 1139,
    "anomaly_start": 40,
    "anomaly_end": 104,
    "anomaly_class": "ego: lateral",
    "num_frames": 140,
    "subset": "val"
  },
  "OWtbKblBOKI_001918": {
    "video_start": 1918,
    "video_end": 1998,
    "anomaly_start": 23,
    "anomaly_end": 49,
    "anomaly_class": "other: lateral",
    "num_frames": 81,
    "subset": "val"
  },
  "OWtbKblBOKI_002459": {
    "video_start": 2459,
    "video_end": 2551,
    "anomaly_start": 29,
    "anomaly_end": 71,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 93,
    "subset": "val"
  },
  "OWtbKblBOKI_003826": {
    "video_start": 3826,
    "video_end": 3920,
    "anomaly_start": 48,
    "anomaly_end": 78,
    "anomaly_class": "ego: oncoming",
    "num_frames": 95,
    "subset": "val"
  },
  "OWtbKblBOKI_004320": {
    "video_start": 4320,
    "video_end": 4387,
    "anomaly_start": 15,
    "anomaly_end": 30,
    "anomaly_class": "ego: oncoming",
    "num_frames": 68,
    "subset": "val"
  },
  "OWtbKblBOKI_004579": {
    "video_start": 4579,
    "video_end": 4698,
    "anomaly_start": 30,
    "anomaly_end": 108,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 120,
    "subset": "val"
  },
  "OWtbKblBOKI_005062": {
    "video_start": 5062,
    "video_end": 5116,
    "anomaly_start": 9,
    "anomaly_end": 55,
    "anomaly_class": "ego: turning",
    "num_frames": 55,
    "subset": "val"
  },
  "OWtbKblBOKI_005256": {
    "video_start": 5256,
    "video_end": 5312,
    "anomaly_start": 25,
    "anomaly_end": 49,
    "anomaly_class": "ego: turning",
    "num_frames": 57,
    "subset": "val"
  },
  "OWtbKblBOKI_005784": {
    "video_start": 5784,
    "video_end": 5900,
    "anomaly_start": 62,
    "anomaly_end": 108,
    "anomaly_class": "ego: start_stop_or_stationary",
    "num_frames": 117,
    "subset": "val"
  },
  "OWtbKblBOKI_005964": {
    "video_start": 5964,
    "video_end": 6044,
    "anomaly_start": 30,
    "anomaly_end": 58,
    "anomaly_class": "ego: turning",
    "num_frames": 81,
    "subset": "val"
  },
  "OWtbKblBOKI_006046": {
    "video_start": 6046,
    "video_end": 6126,
    "anomaly_start": 36,
    "anomaly_end": 81,
    "anomaly_class": "ego: turning",
    "num_frames": 81,
    "subset": "val"
  },
  "P5MjgFmqouA_000823": {
    "video_start": 823,
    "video_end": 931,
    "anomaly_start": 41,
    "anomaly_end": 69,
    "anomaly_class": "ego: lateral",
    "num_frames": 109,
    "subset": "val"
  },
  "P5MjgFmqouA_001306": {
    "video_start": 1306,
    "video_end": 1404,
    "anomaly_start": 44,
    "anomaly_end": 66,
    "anomaly_class": "other: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "P5MjgFmqouA_001406": {
    "video_start": 1406,
    "video_end": 1504,
    "anomaly_start": 42,
    "anomaly_end": 59,
    "anomaly_class": "other: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "P5MjgFmqouA_004030": {
    "video_start": 4030,
    "video_end": 4083,
    "anomaly_start": 46,
    "anomaly_end": 54,
    "anomaly_class": "ego: oncoming",
    "num_frames": 54,
    "subset": "val"
  },
  "PEwiwzyTjX0_000589": {
    "video_start": 589,
    "video_end": 662,
    "anomaly_start": 2,
    "anomaly_end": 45,
    "anomaly_class": "ego: lateral",
    "num_frames": 74,
    "subset": "val"
  },
  "PEwiwzyTjX0_001067": {
    "video_start": 1067,
    "video_end": 1167,
    "anomaly_start": 36,
    "anomaly_end": 70,
    "anomaly_class": "ego: turning",
    "num_frames": 101,
    "subset": "val"
  },
  "PEwiwzyTjX0_002297": {
    "video_start": 2297,
    "video_end": 2378,
    "anomaly_start": 21,
    "anomaly_end": 80,
    "anomaly_class": "ego: lateral",
    "num_frames": 82,
    "subset": "val"
  },
  "PEwiwzyTjX0_002502": {
    "video_start": 2502,
    "video_end": 2590,
    "anomaly_start": 26,
    "anomaly_end": 40,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "PEwiwzyTjX0_004230": {
    "video_start": 4230,
    "video_end": 4359,
    "anomaly_start": 57,
    "anomaly_end": 123,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 130,
    "subset": "val"
  },
  "PEwiwzyTjX0_004544": {
    "video_start": 4544,
    "video_end": 4618,
    "anomaly_start": 29,
    "anomaly_end": 51,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 75,
    "subset": "val"
  },
  "PEwiwzyTjX0_004681": {
    "video_start": 4681,
    "video_end": 4738,
    "anomaly_start": 25,
    "anomaly_end": 36,
    "anomaly_class": "other: turning",
    "num_frames": 58,
    "subset": "val"
  },
  "PEwiwzyTjX0_004740": {
    "video_start": 4740,
    "video_end": 4874,
    "anomaly_start": 46,
    "anomaly_end": 63,
    "anomaly_class": "ego: turning",
    "num_frames": 135,
    "subset": "val"
  },
  "PYL3JcSsS6o_000191": {
    "video_start": 191,
    "video_end": 309,
    "anomaly_start": 71,
    "anomaly_end": 105,
    "anomaly_class": "ego: turning",
    "num_frames": 119,
    "subset": "val"
  },
  "PYL3JcSsS6o_000311": {
    "video_start": 311,
    "video_end": 419,
    "anomaly_start": 57,
    "anomaly_end": 76,
    "anomaly_class": "ego: lateral",
    "num_frames": 109,
    "subset": "val"
  },
  "PYL3JcSsS6o_000862": {
    "video_start": 862,
    "video_end": 960,
    "anomaly_start": 31,
    "anomaly_end": 56,
    "anomaly_class": "other: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "PYL3JcSsS6o_001722": {
    "video_start": 1722,
    "video_end": 1820,
    "anomaly_start": 32,
    "anomaly_end": 53,
    "anomaly_class": "other: oncoming",
    "num_frames": 99,
    "subset": "val"
  },
  "PYL3JcSsS6o_001822": {
    "video_start": 1822,
    "video_end": 1925,
    "anomaly_start": 31,
    "anomaly_end": 63,
    "anomaly_class": "ego: oncoming",
    "num_frames": 104,
    "subset": "val"
  },
  "PYL3JcSsS6o_002428": {
    "video_start": 2428,
    "video_end": 2596,
    "anomaly_start": 38,
    "anomaly_end": 81,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 169,
    "subset": "val"
  },
  "PYL3JcSsS6o_002598": {
    "video_start": 2598,
    "video_end": 2736,
    "anomaly_start": 63,
    "anomaly_end": 88,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 139,
    "subset": "val"
  },
  "PYL3JcSsS6o_004036": {
    "video_start": 4036,
    "video_end": 4144,
    "anomaly_start": 38,
    "anomaly_end": 64,
    "anomaly_class": "other: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "PYL3JcSsS6o_005167": {
    "video_start": 5167,
    "video_end": 5255,
    "anomaly_start": 32,
    "anomaly_end": 64,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 89,
    "subset": "val"
  },
  "PYL3JcSsS6o_005367": {
    "video_start": 5367,
    "video_end": 5455,
    "anomaly_start": 32,
    "anomaly_end": 55,
    "anomaly_class": "other: lateral",
    "num_frames": 89,
    "subset": "val"
  },
  "Pbw0A-RCjcw_000441": {
    "video_start": 441,
    "video_end": 560,
    "anomaly_start": 48,
    "anomaly_end": 88,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 120,
    "subset": "val"
  },
  "Pbw0A-RCjcw_000787": {
    "video_start": 787,
    "video_end": 925,
    "anomaly_start": 57,
    "anomaly_end": 101,
    "anomaly_class": "other: turning",
    "num_frames": 139,
    "subset": "val"
  },
  "Pbw0A-RCjcw_000927": {
    "video_start": 927,
    "video_end": 1028,
    "anomaly_start": 46,
    "anomaly_end": 62,
    "anomaly_class": "other: turning",
    "num_frames": 102,
    "subset": "val"
  },
  "Pbw0A-RCjcw_001030": {
    "video_start": 1030,
    "video_end": 1168,
    "anomaly_start": 54,
    "anomaly_end": 79,
    "anomaly_class": "other: turning",
    "num_frames": 139,
    "subset": "val"
  },
  "Pbw0A-RCjcw_001580": {
    "video_start": 1580,
    "video_end": 1628,
    "anomaly_start": 12,
    "anomaly_end": 49,
    "anomaly_class": "other: pedestrian",
    "num_frames": 49,
    "subset": "val"
  },
  "Pbw0A-RCjcw_002102": {
    "video_start": 2102,
    "video_end": 2210,
    "anomaly_start": 34,
    "anomaly_end": 64,
    "anomaly_class": "ego: oncoming",
    "num_frames": 109,
    "subset": "val"
  },
  "Pbw0A-RCjcw_002642": {
    "video_start": 2642,
    "video_end": 2740,
    "anomaly_start": 37,
    "anomaly_end": 81,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 99,
    "subset": "val"
  },
  "Pbw0A-RCjcw_002938": {
    "video_start": 2938,
    "video_end": 3056,
    "anomaly_start": 41,
    "anomaly_end": 82,
    "anomaly_class": "other: turning",
    "num_frames": 119,
    "subset": "val"
  },
  "PfJ2CudpCgE_000927": {
    "video_start": 927,
    "video_end": 1037,
    "anomaly_start": 40,
    "anomaly_end": 85,
    "anomaly_class": "ego: lateral",
    "num_frames": 111,
    "subset": "val"
  },
  "PfJ2CudpCgE_002202": {
    "video_start": 2202,
    "video_end": 2275,
    "anomaly_start": 30,
    "anomaly_end": 48,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 74,
    "subset": "val"
  },
  "PfJ2CudpCgE_002955": {
    "video_start": 2955,
    "video_end": 3065,
    "anomaly_start": 56,
    "anomaly_end": 70,
    "anomaly_class": "ego: turning",
    "num_frames": 111,
    "subset": "val"
  },
  "PfJ2CudpCgE_003067": {
    "video_start": 3067,
    "video_end": 3152,
    "anomaly_start": 31,
    "anomaly_end": 70,
    "anomaly_class": "ego: lateral",
    "num_frames": 86,
    "subset": "val"
  },
  "PfJ2CudpCgE_003409": {
    "video_start": 3409,
    "video_end": 3504,
    "anomaly_start": 61,
    "anomaly_end": 80,
    "anomaly_class": "ego: unknown",
    "num_frames": 96,
    "subset": "val"
  },
  "PfJ2CudpCgE_003506": {
    "video_start": 3506,
    "video_end": 3612,
    "anomaly_start": 66,
    "anomaly_end": 88,
    "anomaly_class": "ego: lateral",
    "num_frames": 107,
    "subset": "val"
  },
  "PfJ2CudpCgE_003831": {
    "video_start": 3831,
    "video_end": 3922,
    "anomaly_start": 35,
    "anomaly_end": 73,
    "anomaly_class": "ego: turning",
    "num_frames": 92,
    "subset": "val"
  },
  "PfJ2CudpCgE_004303": {
    "video_start": 4303,
    "video_end": 4393,
    "anomaly_start": 46,
    "anomaly_end": 73,
    "anomaly_class": "ego: turning",
    "num_frames": 91,
    "subset": "val"
  },
  "PfJ2CudpCgE_004728": {
    "video_start": 4728,
    "video_end": 4814,
    "anomaly_start": 43,
    "anomaly_end": 81,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 87,
    "subset": "val"
  },
  "PfJ2CudpCgE_004816": {
    "video_start": 4816,
    "video_end": 4931,
    "anomaly_start": 25,
    "anomaly_end": 99,
    "anomaly_class": "other: obstacle",
    "num_frames": 116,
    "subset": "val"
  },
  "PfJ2CudpCgE_005574": {
    "video_start": 5574,
    "video_end": 5654,
    "anomaly_start": 44,
    "anomaly_end": 74,
    "anomaly_class": "ego: turning",
    "num_frames": 81,
    "subset": "val"
  },
  "PfJ2CudpCgE_005947": {
    "video_start": 5947,
    "video_end": 6022,
    "anomaly_start": 28,
    "anomaly_end": 56,
    "anomaly_class": "other: turning",
    "num_frames": 76,
    "subset": "val"
  },
  "PqbpIHZvjMA_001464": {
    "video_start": 1464,
    "video_end": 1578,
    "anomaly_start": 27,
    "anomaly_end": 63,
    "anomaly_class": "other: turning",
    "num_frames": 115,
    "subset": "val"
  },
  "PqbpIHZvjMA_001933": {
    "video_start": 1933,
    "video_end": 2039,
    "anomaly_start": 22,
    "anomaly_end": 32,
    "anomaly_class": "ego: lateral",
    "num_frames": 107,
    "subset": "val"
  },
  "PqbpIHZvjMA_002926": {
    "video_start": 2926,
    "video_end": 3044,
    "anomaly_start": 34,
    "anomaly_end": 86,
    "anomaly_class": "other: oncoming",
    "num_frames": 119,
    "subset": "val"
  },
  "PqbpIHZvjMA_003046": {
    "video_start": 3046,
    "video_end": 3184,
    "anomaly_start": 75,
    "anomaly_end": 89,
    "anomaly_class": "other: start_stop_or_stationary",
    "num_frames": 139,
    "subset": "val"
  },
  "PqbpIHZvjMA_003186": {
    "video_start": 3186,
    "video_end": 3365,
    "anomaly_start": 36,
    "anomaly_end": 60,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 180,
    "subset": "val"
  },
  "PqbpIHZvjMA_003367": {
    "video_start": 3367,
    "video_end": 3465,
    "anomaly_start": 34,
    "anomaly_end": 99,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 99,
    "subset": "val"
  },
  "PqbpIHZvjMA_004933": {
    "video_start": 4933,
    "video_end": 4988,
    "anomaly_start": 45,
    "anomaly_end": 56,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 56,
    "subset": "val"
  },
  "PqbpIHZvjMA_005165": {
    "video_start": 5165,
    "video_end": 5272,
    "anomaly_start": 24,
    "anomaly_end": 72,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 108,
    "subset": "val"
  },
  "PqbpIHZvjMA_005274": {
    "video_start": 5274,
    "video_end": 5362,
    "anomaly_start": 42,
    "anomaly_end": 71,
    "anomaly_class": "ego: lateral",
    "num_frames": 89,
    "subset": "val"
  },
  "Q7VBPeGwJWw_000051": {
    "video_start": 51,
    "video_end": 155,
    "anomaly_start": 31,
    "anomaly_end": 76,
    "anomaly_class": "ego: turning",
    "num_frames": 105,
    "subset": "val"
  },
  "Q7VBPeGwJWw_000157": {
    "video_start": 157,
    "video_end": 259,
    "anomaly_start": 22,
    "anomaly_end": 48,
    "anomaly_class": "other: turning",
    "num_frames": 103,
    "subset": "val"
  },
  "Q7VBPeGwJWw_000261": {
    "video_start": 261,
    "video_end": 355,
    "anomaly_start": 31,
    "anomaly_end": 74,
    "anomaly_class": "ego: turning",
    "num_frames": 95,
    "subset": "val"
  },
  "Q7VBPeGwJWw_000938": {
    "video_start": 938,
    "video_end": 1021,
    "anomaly_start": 11,
    "anomaly_end": 40,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 84,
    "subset": "val"
  },
  "Q7VBPeGwJWw_001409": {
    "video_start": 1409,
    "video_end": 1503,
    "anomaly_start": 36,
    "anomaly_end": 57,
    "anomaly_class": "ego: turning",
    "num_frames": 95,
    "subset": "val"
  },
  "Q7VBPeGwJWw_001505": {
    "video_start": 1505,
    "video_end": 1629,
    "anomaly_start": 19,
    "anomaly_end": 102,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 125,
    "subset": "val"
  },
  "Q7VBPeGwJWw_001631": {
    "video_start": 1631,
    "video_end": 1701,
    "anomaly_start": 29,
    "anomaly_end": 62,
    "anomaly_class": "other: obstacle",
    "num_frames": 71,
    "subset": "val"
  },
  "Q7VBPeGwJWw_001856": {
    "video_start": 1856,
    "video_end": 1962,
    "anomaly_start": 40,
    "anomaly_end": 72,
    "anomaly_class": "other: turning",
    "num_frames": 107,
    "subset": "val"
  },
  "Q7VBPeGwJWw_002331": {
    "video_start": 2331,
    "video_end": 2429,
    "anomaly_start": 36,
    "anomaly_end": 61,
    "anomaly_class": "ego: lateral",
    "num_frames": 99,
    "subset": "val"
  },
  "Q7VBPeGwJWw_002764": {
    "video_start": 2764,
    "video_end": 2847,
    "anomaly_start": 25,
    "anomaly_end": 47,
    "anomaly_class": "other: turning",
    "num_frames": 84,
    "subset": "val"
  },
  "QERvirE3S5s_000052": {
    "video_start": 52,
    "video_end": 185,
    "anomaly_start": 25,
    "anomaly_end": 100,
    "anomaly_class": "other: obstacle",
    "num_frames": 134,
    "subset": "val"
  },
  "QERvirE3S5s_000654": {
    "video_start": 654,
    "video_end": 742,
    "anomaly_start": 25,
    "anomaly_end": 62,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "QERvirE3S5s_001573": {
    "video_start": 1573,
    "video_end": 1662,
    "anomaly_start": 25,
    "anomaly_end": 75,
    "anomaly_class": "ego: lateral",
    "num_frames": 90,
    "subset": "val"
  },
  "QERvirE3S5s_002162": {
    "video_start": 2162,
    "video_end": 2302,
    "anomaly_start": 30,
    "anomaly_end": 61,
    "anomaly_class": "ego: lateral",
    "num_frames": 141,
    "subset": "val"
  },
  "QERvirE3S5s_003568": {
    "video_start": 3568,
    "video_end": 3671,
    "anomaly_start": 49,
    "anomaly_end": 82,
    "anomaly_class": "ego: turning",
    "num_frames": 104,
    "subset": "val"
  },
  "QERvirE3S5s_004537": {
    "video_start": 4537,
    "video_end": 4616,
    "anomaly_start": 20,
    "anomaly_end": 41,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 80,
    "subset": "val"
  },
  "QERvirE3S5s_005255": {
    "video_start": 5255,
    "video_end": 5348,
    "anomaly_start": 32,
    "anomaly_end": 94,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 94,
    "subset": "val"
  },
  "QERvirE3S5s_005834": {
    "video_start": 5834,
    "video_end": 5931,
    "anomaly_start": 66,
    "anomaly_end": 82,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 98,
    "subset": "val"
  },
  "QeuzUfHIWYU_000469": {
    "video_start": 469,
    "video_end": 557,
    "anomaly_start": 36,
    "anomaly_end": 48,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "QeuzUfHIWYU_001416": {
    "video_start": 1416,
    "video_end": 1515,
    "anomaly_start": 28,
    "anomaly_end": 57,
    "anomaly_class": "other: turning",
    "num_frames": 100,
    "subset": "val"
  },
  "QeuzUfHIWYU_002767": {
    "video_start": 2767,
    "video_end": 2876,
    "anomaly_start": 48,
    "anomaly_end": 86,
    "anomaly_class": "ego: lateral",
    "num_frames": 110,
    "subset": "val"
  },
  "RASKiMoxhOE_000246": {
    "video_start": 246,
    "video_end": 354,
    "anomaly_start": 36,
    "anomaly_end": 61,
    "anomaly_class": "ego: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "RASKiMoxhOE_000542": {
    "video_start": 542,
    "video_end": 620,
    "anomaly_start": 46,
    "anomaly_end": 79,
    "anomaly_class": "ego: oncoming",
    "num_frames": 79,
    "subset": "val"
  },
  "RASKiMoxhOE_001588": {
    "video_start": 1588,
    "video_end": 1669,
    "anomaly_start": 32,
    "anomaly_end": 58,
    "anomaly_class": "ego: lateral",
    "num_frames": 82,
    "subset": "val"
  },
  "RASKiMoxhOE_001671": {
    "video_start": 1671,
    "video_end": 1748,
    "anomaly_start": 28,
    "anomaly_end": 50,
    "anomaly_class": "ego: lateral",
    "num_frames": 78,
    "subset": "val"
  },
  "RASKiMoxhOE_001821": {
    "video_start": 1821,
    "video_end": 1895,
    "anomaly_start": 27,
    "anomaly_end": 52,
    "anomaly_class": "ego: turning",
    "num_frames": 75,
    "subset": "val"
  },
  "RASKiMoxhOE_001958": {
    "video_start": 1958,
    "video_end": 2042,
    "anomaly_start": 30,
    "anomaly_end": 46,
    "anomaly_class": "other: turning",
    "num_frames": 85,
    "subset": "val"
  },
  "RASKiMoxhOE_002044": {
    "video_start": 2044,
    "video_end": 2126,
    "anomaly_start": 22,
    "anomaly_end": 51,
    "anomaly_class": "ego: turning",
    "num_frames": 83,
    "subset": "val"
  },
  "RASKiMoxhOE_002350": {
    "video_start": 2350,
    "video_end": 2495,
    "anomaly_start": 35,
    "anomaly_end": 50,
    "anomaly_class": "ego: lateral",
    "num_frames": 146,
    "subset": "val"
  },
  "RASKiMoxhOE_002656": {
    "video_start": 2656,
    "video_end": 2730,
    "anomaly_start": 23,
    "anomaly_end": 44,
    "anomaly_class": "ego: turning",
    "num_frames": 75,
    "subset": "val"
  },
  "RASKiMoxhOE_002732": {
    "video_start": 2732,
    "video_end": 2807,
    "anomaly_start": 33,
    "anomaly_end": 47,
    "anomaly_class": "ego: turning",
    "num_frames": 76,
    "subset": "val"
  },
  "RASKiMoxhOE_002809": {
    "video_start": 2809,
    "video_end": 2906,
    "anomaly_start": 29,
    "anomaly_end": 64,
    "anomaly_class": "ego: oncoming",
    "num_frames": 98,
    "subset": "val"
  },
  "RASKiMoxhOE_002908": {
    "video_start": 2908,
    "video_end": 3018,
    "anomaly_start": 29,
    "anomaly_end": 73,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 111,
    "subset": "val"
  },
  "RASKiMoxhOE_003020": {
    "video_start": 3020,
    "video_end": 3102,
    "anomaly_start": 41,
    "anomaly_end": 60,
    "anomaly_class": "ego: oncoming",
    "num_frames": 83,
    "subset": "val"
  },
  "RASKiMoxhOE_004422": {
    "video_start": 4422,
    "video_end": 4500,
    "anomaly_start": 25,
    "anomaly_end": 61,
    "anomaly_class": "ego: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "RASKiMoxhOE_004705": {
    "video_start": 4705,
    "video_end": 4820,
    "anomaly_start": 41,
    "anomaly_end": 92,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 116,
    "subset": "val"
  },
  "RASKiMoxhOE_005128": {
    "video_start": 5128,
    "video_end": 5200,
    "anomaly_start": 21,
    "anomaly_end": 60,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 73,
    "subset": "val"
  },
  "RASKiMoxhOE_005417": {
    "video_start": 5417,
    "video_end": 5458,
    "anomaly_start": 21,
    "anomaly_end": 42,
    "anomaly_class": "ego: lateral",
    "num_frames": 42,
    "subset": "val"
  },
  "RASKiMoxhOE_005460": {
    "video_start": 5460,
    "video_end": 5560,
    "anomaly_start": 17,
    "anomaly_end": 59,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 101,
    "subset": "val"
  },
  "RASKiMoxhOE_005562": {
    "video_start": 5562,
    "video_end": 5631,
    "anomaly_start": 33,
    "anomaly_end": 52,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 70,
    "subset": "val"
  },
  "RPNiqsdt6-w_000611": {
    "video_start": 611,
    "video_end": 686,
    "anomaly_start": 71,
    "anomaly_end": 76,
    "anomaly_class": "ego: oncoming",
    "num_frames": 76,
    "subset": "val"
  },
  "RPNiqsdt6-w_001284": {
    "video_start": 1284,
    "video_end": 1422,
    "anomaly_start": 93,
    "anomaly_end": 118,
    "anomaly_class": "ego: turning",
    "num_frames": 139,
    "subset": "val"
  },
  "RPNiqsdt6-w_001424": {
    "video_start": 1424,
    "video_end": 1562,
    "anomaly_start": 50,
    "anomaly_end": 124,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 139,
    "subset": "val"
  },
  "RPNiqsdt6-w_001684": {
    "video_start": 1684,
    "video_end": 1802,
    "anomaly_start": 19,
    "anomaly_end": 44,
    "anomaly_class": "other: oncoming",
    "num_frames": 119,
    "subset": "val"
  },
  "RPNiqsdt6-w_003187": {
    "video_start": 3187,
    "video_end": 3335,
    "anomaly_start": 56,
    "anomaly_end": 106,
    "anomaly_class": "ego: lateral",
    "num_frames": 149,
    "subset": "val"
  },
  "RPNiqsdt6-w_003547": {
    "video_start": 3547,
    "video_end": 3645,
    "anomaly_start": 38,
    "anomaly_end": 62,
    "anomaly_class": "other: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "RPNiqsdt6-w_004024": {
    "video_start": 4024,
    "video_end": 4088,
    "anomaly_start": 56,
    "anomaly_end": 65,
    "anomaly_class": "ego: turning",
    "num_frames": 65,
    "subset": "val"
  },
  "RPNiqsdt6-w_004114": {
    "video_start": 4114,
    "video_end": 4173,
    "anomaly_start": 52,
    "anomaly_end": 60,
    "anomaly_class": "ego: turning",
    "num_frames": 60,
    "subset": "val"
  },
  "RPNiqsdt6-w_004314": {
    "video_start": 4314,
    "video_end": 4432,
    "anomaly_start": 44,
    "anomaly_end": 99,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 119,
    "subset": "val"
  },
  "RPNiqsdt6-w_004642": {
    "video_start": 4642,
    "video_end": 4780,
    "anomaly_start": 40,
    "anomaly_end": 115,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 139,
    "subset": "val"
  },
  "RPNiqsdt6-w_005025": {
    "video_start": 5025,
    "video_end": 5133,
    "anomaly_start": 40,
    "anomaly_end": 79,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 109,
    "subset": "val"
  },
  "RPNiqsdt6-w_005405": {
    "video_start": 5405,
    "video_end": 5506,
    "anomaly_start": 22,
    "anomaly_end": 73,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 102,
    "subset": "val"
  },
  "Rrcjy5t8b20_000352": {
    "video_start": 352,
    "video_end": 475,
    "anomaly_start": 27,
    "anomaly_end": 61,
    "anomaly_class": "ego: start_stop_or_stationary",
    "num_frames": 124,
    "subset": "val"
  },
  "Rrcjy5t8b20_000907": {
    "video_start": 907,
    "video_end": 1004,
    "anomaly_start": 46,
    "anomaly_end": 70,
    "anomaly_class": "ego: lateral",
    "num_frames": 98,
    "subset": "val"
  },
  "Rrcjy5t8b20_001006": {
    "video_start": 1006,
    "video_end": 1096,
    "anomaly_start": 31,
    "anomaly_end": 49,
    "anomaly_class": "ego: turning",
    "num_frames": 91,
    "subset": "val"
  },
  "Rrcjy5t8b20_001187": {
    "video_start": 1187,
    "video_end": 1270,
    "anomaly_start": 30,
    "anomaly_end": 52,
    "anomaly_class": "ego: turning",
    "num_frames": 84,
    "subset": "val"
  },
  "Rrcjy5t8b20_002784": {
    "video_start": 2784,
    "video_end": 2995,
    "anomaly_start": 26,
    "anomaly_end": 44,
    "anomaly_class": "ego: oncoming",
    "num_frames": 212,
    "subset": "val"
  },
  "Rrcjy5t8b20_003448": {
    "video_start": 3448,
    "video_end": 3496,
    "anomaly_start": 10,
    "anomaly_end": 20,
    "anomaly_class": "ego: turning",
    "num_frames": 49,
    "subset": "val"
  },
  "Rrcjy5t8b20_003828": {
    "video_start": 3828,
    "video_end": 3918,
    "anomaly_start": 25,
    "anomaly_end": 82,
    "anomaly_class": "ego: lateral",
    "num_frames": 91,
    "subset": "val"
  },
  "Rrcjy5t8b20_004212": {
    "video_start": 4212,
    "video_end": 4285,
    "anomaly_start": 35,
    "anomaly_end": 48,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 74,
    "subset": "val"
  },
  "Sihe6aeyLHg_000052": {
    "video_start": 52,
    "video_end": 153,
    "anomaly_start": 28,
    "anomaly_end": 48,
    "anomaly_class": "ego: turning",
    "num_frames": 102,
    "subset": "val"
  },
  "Sihe6aeyLHg_000242": {
    "video_start": 242,
    "video_end": 349,
    "anomaly_start": 63,
    "anomaly_end": 82,
    "anomaly_class": "ego: start_stop_or_stationary",
    "num_frames": 108,
    "subset": "val"
  },
  "Sihe6aeyLHg_000602": {
    "video_start": 602,
    "video_end": 657,
    "anomaly_start": 45,
    "anomaly_end": 56,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 56,
    "subset": "val"
  },
  "Sihe6aeyLHg_001050": {
    "video_start": 1050,
    "video_end": 1137,
    "anomaly_start": 40,
    "anomaly_end": 74,
    "anomaly_class": "ego: turning",
    "num_frames": 88,
    "subset": "val"
  },
  "Sihe6aeyLHg_001139": {
    "video_start": 1139,
    "video_end": 1233,
    "anomaly_start": 60,
    "anomaly_end": 81,
    "anomaly_class": "ego: turning",
    "num_frames": 95,
    "subset": "val"
  },
  "Sihe6aeyLHg_001592": {
    "video_start": 1592,
    "video_end": 1678,
    "anomaly_start": 27,
    "anomaly_end": 56,
    "anomaly_class": "ego: turning",
    "num_frames": 87,
    "subset": "val"
  },
  "Sihe6aeyLHg_001779": {
    "video_start": 1779,
    "video_end": 1857,
    "anomaly_start": 28,
    "anomaly_end": 63,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 79,
    "subset": "val"
  },
  "Sihe6aeyLHg_002266": {
    "video_start": 2266,
    "video_end": 2364,
    "anomaly_start": 45,
    "anomaly_end": 66,
    "anomaly_class": "ego: lateral",
    "num_frames": 99,
    "subset": "val"
  },
  "Sihe6aeyLHg_003418": {
    "video_start": 3418,
    "video_end": 3524,
    "anomaly_start": 15,
    "anomaly_end": 35,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 107,
    "subset": "val"
  },
  "Sihe6aeyLHg_004435": {
    "video_start": 4435,
    "video_end": 4546,
    "anomaly_start": 61,
    "anomaly_end": 90,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 112,
    "subset": "val"
  },
  "Sihe6aeyLHg_005275": {
    "video_start": 5275,
    "video_end": 5350,
    "anomaly_start": 36,
    "anomaly_end": 61,
    "anomaly_class": "other: turning",
    "num_frames": 76,
    "subset": "val"
  },
  "Sihe6aeyLHg_005352": {
    "video_start": 5352,
    "video_end": 5428,
    "anomaly_start": 43,
    "anomaly_end": 51,
    "anomaly_class": "ego: turning",
    "num_frames": 77,
    "subset": "val"
  },
  "Sihe6aeyLHg_005590": {
    "video_start": 5590,
    "video_end": 5674,
    "anomaly_start": 36,
    "anomaly_end": 65,
    "anomaly_class": "ego: turning",
    "num_frames": 85,
    "subset": "val"
  },
  "T7TkJVmGyts_000431": {
    "video_start": 431,
    "video_end": 501,
    "anomaly_start": 26,
    "anomaly_end": 71,
    "anomaly_class": "ego: turning",
    "num_frames": 71,
    "subset": "val"
  },
  "T7TkJVmGyts_001011": {
    "video_start": 1011,
    "video_end": 1132,
    "anomaly_start": 31,
    "anomaly_end": 55,
    "anomaly_class": "other: turning",
    "num_frames": 122,
    "subset": "val"
  },
  "T7TkJVmGyts_001508": {
    "video_start": 1508,
    "video_end": 1626,
    "anomaly_start": 69,
    "anomaly_end": 89,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 119,
    "subset": "val"
  },
  "T7TkJVmGyts_002905": {
    "video_start": 2905,
    "video_end": 3023,
    "anomaly_start": 51,
    "anomaly_end": 63,
    "anomaly_class": "other: obstacle",
    "num_frames": 119,
    "subset": "val"
  },
  "T7TkJVmGyts_003025": {
    "video_start": 3025,
    "video_end": 3090,
    "anomaly_start": 53,
    "anomaly_end": 66,
    "anomaly_class": "ego: oncoming",
    "num_frames": 66,
    "subset": "val"
  },
  "T7TkJVmGyts_003265": {
    "video_start": 3265,
    "video_end": 3391,
    "anomaly_start": 58,
    "anomaly_end": 99,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 127,
    "subset": "val"
  },
  "T7TkJVmGyts_003713": {
    "video_start": 3713,
    "video_end": 3831,
    "anomaly_start": 29,
    "anomaly_end": 46,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 119,
    "subset": "val"
  },
  "T7TkJVmGyts_003933": {
    "video_start": 3933,
    "video_end": 4021,
    "anomaly_start": 36,
    "anomaly_end": 64,
    "anomaly_class": "other: obstacle",
    "num_frames": 89,
    "subset": "val"
  },
  "T7TkJVmGyts_004023": {
    "video_start": 4023,
    "video_end": 4141,
    "anomaly_start": 27,
    "anomaly_end": 67,
    "anomaly_class": "other: turning",
    "num_frames": 119,
    "subset": "val"
  },
  "T7TkJVmGyts_004143": {
    "video_start": 4143,
    "video_end": 4261,
    "anomaly_start": 17,
    "anomaly_end": 83,
    "anomaly_class": "other: turning",
    "num_frames": 119,
    "subset": "val"
  },
  "T89zSiaMNgM_000182": {
    "video_start": 182,
    "video_end": 285,
    "anomaly_start": 26,
    "anomaly_end": 90,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 104,
    "subset": "val"
  },
  "T89zSiaMNgM_000287": {
    "video_start": 287,
    "video_end": 405,
    "anomaly_start": 37,
    "anomaly_end": 104,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 119,
    "subset": "val"
  },
  "T89zSiaMNgM_000783": {
    "video_start": 783,
    "video_end": 867,
    "anomaly_start": 54,
    "anomaly_end": 85,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 85,
    "subset": "val"
  },
  "T89zSiaMNgM_001095": {
    "video_start": 1095,
    "video_end": 1203,
    "anomaly_start": 34,
    "anomaly_end": 92,
    "anomaly_class": "ego: oncoming",
    "num_frames": 109,
    "subset": "val"
  },
  "T89zSiaMNgM_001205": {
    "video_start": 1205,
    "video_end": 1293,
    "anomaly_start": 45,
    "anomaly_end": 60,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "T89zSiaMNgM_001989": {
    "video_start": 1989,
    "video_end": 2113,
    "anomaly_start": 44,
    "anomaly_end": 96,
    "anomaly_class": "other: turning",
    "num_frames": 125,
    "subset": "val"
  },
  "T89zSiaMNgM_004104": {
    "video_start": 4104,
    "video_end": 4192,
    "anomaly_start": 18,
    "anomaly_end": 33,
    "anomaly_class": "other: lateral",
    "num_frames": 89,
    "subset": "val"
  },
  "T89zSiaMNgM_004605": {
    "video_start": 4605,
    "video_end": 4713,
    "anomaly_start": 46,
    "anomaly_end": 57,
    "anomaly_class": "other: lateral",
    "num_frames": 109,
    "subset": "val"
  },
  "T89zSiaMNgM_004818": {
    "video_start": 4818,
    "video_end": 4907,
    "anomaly_start": 37,
    "anomaly_end": 58,
    "anomaly_class": "ego: pedestrian",
    "num_frames": 90,
    "subset": "val"
  },
  "T89zSiaMNgM_005373": {
    "video_start": 5373,
    "video_end": 5464,
    "anomaly_start": 59,
    "anomaly_end": 75,
    "anomaly_class": "other: obstacle",
    "num_frames": 92,
    "subset": "val"
  },
  "TNZv-NBcV5U_000066": {
    "video_start": 66,
    "video_end": 154,
    "anomaly_start": 35,
    "anomaly_end": 56,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 89,
    "subset": "val"
  },
  "TNZv-NBcV5U_000472": {
    "video_start": 472,
    "video_end": 551,
    "anomaly_start": 38,
    "anomaly_end": 53,
    "anomaly_class": "ego: turning",
    "num_frames": 80,
    "subset": "val"
  },
  "TNZv-NBcV5U_000553": {
    "video_start": 553,
    "video_end": 626,
    "anomaly_start": 39,
    "anomaly_end": 60,
    "anomaly_class": "other: pedestrian",
    "num_frames": 74,
    "subset": "val"
  },
  "TNZv-NBcV5U_001353": {
    "video_start": 1353,
    "video_end": 1441,
    "anomaly_start": 30,
    "anomaly_end": 56,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "TNZv-NBcV5U_001918": {
    "video_start": 1918,
    "video_end": 2000,
    "anomaly_start": 14,
    "anomaly_end": 52,
    "anomaly_class": "ego: turning",
    "num_frames": 83,
    "subset": "val"
  },
  "TNZv-NBcV5U_002389": {
    "video_start": 2389,
    "video_end": 2493,
    "anomaly_start": 26,
    "anomaly_end": 50,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 105,
    "subset": "val"
  },
  "TNZv-NBcV5U_002495": {
    "video_start": 2495,
    "video_end": 2570,
    "anomaly_start": 51,
    "anomaly_end": 75,
    "anomaly_class": "other: turning",
    "num_frames": 76,
    "subset": "val"
  },
  "TNZv-NBcV5U_002660": {
    "video_start": 2660,
    "video_end": 2766,
    "anomaly_start": 45,
    "anomaly_end": 68,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 107,
    "subset": "val"
  },
  "TNZv-NBcV5U_002856": {
    "video_start": 2856,
    "video_end": 2938,
    "anomaly_start": 9,
    "anomaly_end": 41,
    "anomaly_class": "ego: turning",
    "num_frames": 83,
    "subset": "val"
  },
  "TNZv-NBcV5U_002940": {
    "video_start": 2940,
    "video_end": 3038,
    "anomaly_start": 32,
    "anomaly_end": 38,
    "anomaly_class": "ego: lateral",
    "num_frames": 99,
    "subset": "val"
  },
  "TNZv-NBcV5U_003190": {
    "video_start": 3190,
    "video_end": 3251,
    "anomaly_start": 37,
    "anomaly_end": 59,
    "anomaly_class": "ego: turning",
    "num_frames": 62,
    "subset": "val"
  },
  "TNZv-NBcV5U_003253": {
    "video_start": 3253,
    "video_end": 3331,
    "anomaly_start": 29,
    "anomaly_end": 45,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 79,
    "subset": "val"
  },
  "TNZv-NBcV5U_003414": {
    "video_start": 3414,
    "video_end": 3491,
    "anomaly_start": 21,
    "anomaly_end": 45,
    "anomaly_class": "ego: lateral",
    "num_frames": 78,
    "subset": "val"
  },
  "TNZv-NBcV5U_003715": {
    "video_start": 3715,
    "video_end": 3792,
    "anomaly_start": 32,
    "anomaly_end": 54,
    "anomaly_class": "ego: turning",
    "num_frames": 78,
    "subset": "val"
  },
  "TNZv-NBcV5U_003794": {
    "video_start": 3794,
    "video_end": 3936,
    "anomaly_start": 28,
    "anomaly_end": 41,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 143,
    "subset": "val"
  },
  "TNZv-NBcV5U_004086": {
    "video_start": 4086,
    "video_end": 4198,
    "anomaly_start": 40,
    "anomaly_end": 86,
    "anomaly_class": "other: lateral",
    "num_frames": 113,
    "subset": "val"
  },
  "TNZv-NBcV5U_004200": {
    "video_start": 4200,
    "video_end": 4288,
    "anomaly_start": 40,
    "anomaly_end": 54,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "TNZv-NBcV5U_004436": {
    "video_start": 4436,
    "video_end": 4525,
    "anomaly_start": 49,
    "anomaly_end": 70,
    "anomaly_class": "other: pedestrian",
    "num_frames": 90,
    "subset": "val"
  },
  "TNZv-NBcV5U_004604": {
    "video_start": 4604,
    "video_end": 4685,
    "anomaly_start": 42,
    "anomaly_end": 51,
    "anomaly_class": "ego: oncoming",
    "num_frames": 82,
    "subset": "val"
  },
  "TlBhM9Xbr2o_000416": {
    "video_start": 416,
    "video_end": 510,
    "anomaly_start": 37,
    "anomaly_end": 67,
    "anomaly_class": "ego: oncoming",
    "num_frames": 95,
    "subset": "val"
  },
  "TlBhM9Xbr2o_000512": {
    "video_start": 512,
    "video_end": 627,
    "anomaly_start": 36,
    "anomaly_end": 86,
    "anomaly_class": "other: lateral",
    "num_frames": 116,
    "subset": "val"
  },
  "TlBhM9Xbr2o_000629": {
    "video_start": 629,
    "video_end": 721,
    "anomaly_start": 42,
    "anomaly_end": 58,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 93,
    "subset": "val"
  },
  "TlBhM9Xbr2o_000852": {
    "video_start": 852,
    "video_end": 960,
    "anomaly_start": 36,
    "anomaly_end": 57,
    "anomaly_class": "other: lateral",
    "num_frames": 109,
    "subset": "val"
  },
  "TlBhM9Xbr2o_000962": {
    "video_start": 962,
    "video_end": 1090,
    "anomaly_start": 72,
    "anomaly_end": 106,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 129,
    "subset": "val"
  },
  "TlBhM9Xbr2o_002504": {
    "video_start": 2504,
    "video_end": 2588,
    "anomaly_start": 15,
    "anomaly_end": 57,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 85,
    "subset": "val"
  },
  "TlBhM9Xbr2o_003153": {
    "video_start": 3153,
    "video_end": 3231,
    "anomaly_start": 11,
    "anomaly_end": 26,
    "anomaly_class": "other: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "TlBhM9Xbr2o_003692": {
    "video_start": 3692,
    "video_end": 3784,
    "anomaly_start": 21,
    "anomaly_end": 55,
    "anomaly_class": "other: lateral",
    "num_frames": 93,
    "subset": "val"
  },
  "TlBhM9Xbr2o_004105": {
    "video_start": 4105,
    "video_end": 4218,
    "anomaly_start": 31,
    "anomaly_end": 72,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 114,
    "subset": "val"
  },
  "Tsl84N96WM8_000441": {
    "video_start": 441,
    "video_end": 579,
    "anomaly_start": 41,
    "anomaly_end": 54,
    "anomaly_class": "other: turning",
    "num_frames": 139,
    "subset": "val"
  },
  "Tsl84N96WM8_000581": {
    "video_start": 581,
    "video_end": 679,
    "anomaly_start": 60,
    "anomaly_end": 75,
    "anomaly_class": "ego: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "Tsl84N96WM8_001791": {
    "video_start": 1791,
    "video_end": 1930,
    "anomaly_start": 78,
    "anomaly_end": 115,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 140,
    "subset": "val"
  },
  "Tsl84N96WM8_002262": {
    "video_start": 2262,
    "video_end": 2360,
    "anomaly_start": 33,
    "anomaly_end": 68,
    "anomaly_class": "ego: unknown",
    "num_frames": 99,
    "subset": "val"
  },
  "Tsl84N96WM8_002572": {
    "video_start": 2572,
    "video_end": 2690,
    "anomaly_start": 49,
    "anomaly_end": 85,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 119,
    "subset": "val"
  },
  "Tsl84N96WM8_003271": {
    "video_start": 3271,
    "video_end": 3384,
    "anomaly_start": 18,
    "anomaly_end": 74,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 114,
    "subset": "val"
  },
  "Tsl84N96WM8_003867": {
    "video_start": 3867,
    "video_end": 3955,
    "anomaly_start": 36,
    "anomaly_end": 62,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 89,
    "subset": "val"
  },
  "Tsl84N96WM8_003957": {
    "video_start": 3957,
    "video_end": 4055,
    "anomaly_start": 24,
    "anomaly_end": 87,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 99,
    "subset": "val"
  },
  "Tsl84N96WM8_004167": {
    "video_start": 4167,
    "video_end": 4265,
    "anomaly_start": 51,
    "anomaly_end": 82,
    "anomaly_class": "ego: oncoming",
    "num_frames": 99,
    "subset": "val"
  },
  "Tsl84N96WM8_004577": {
    "video_start": 4577,
    "video_end": 4715,
    "anomaly_start": 52,
    "anomaly_end": 104,
    "anomaly_class": "ego: lateral",
    "num_frames": 139,
    "subset": "val"
  },
  "Tsl84N96WM8_004887": {
    "video_start": 4887,
    "video_end": 4965,
    "anomaly_start": 34,
    "anomaly_end": 49,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 79,
    "subset": "val"
  },
  "Tsl84N96WM8_004967": {
    "video_start": 4967,
    "video_end": 5115,
    "anomaly_start": 63,
    "anomaly_end": 99,
    "anomaly_class": "ego: turning",
    "num_frames": 149,
    "subset": "val"
  },
  "Tsl84N96WM8_005117": {
    "video_start": 5117,
    "video_end": 5255,
    "anomaly_start": 46,
    "anomaly_end": 81,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 139,
    "subset": "val"
  },
  "Tsl84N96WM8_005257": {
    "video_start": 5257,
    "video_end": 5381,
    "anomaly_start": 36,
    "anomaly_end": 102,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 125,
    "subset": "val"
  },
  "UKBuDqU8qTM_000080": {
    "video_start": 80,
    "video_end": 189,
    "anomaly_start": 46,
    "anomaly_end": 75,
    "anomaly_class": "ego: turning",
    "num_frames": 110,
    "subset": "val"
  },
  "UKBuDqU8qTM_002630": {
    "video_start": 2630,
    "video_end": 2728,
    "anomaly_start": 42,
    "anomaly_end": 73,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 99,
    "subset": "val"
  },
  "UKBuDqU8qTM_002910": {
    "video_start": 2910,
    "video_end": 3028,
    "anomaly_start": 49,
    "anomaly_end": 94,
    "anomaly_class": "ego: lateral",
    "num_frames": 119,
    "subset": "val"
  },
  "UKBuDqU8qTM_003591": {
    "video_start": 3591,
    "video_end": 3689,
    "anomaly_start": 31,
    "anomaly_end": 49,
    "anomaly_class": "ego: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "UKBuDqU8qTM_003691": {
    "video_start": 3691,
    "video_end": 3813,
    "anomaly_start": 25,
    "anomaly_end": 43,
    "anomaly_class": "other: turning",
    "num_frames": 123,
    "subset": "val"
  },
  "UOjQsfDXjO0_000221": {
    "video_start": 221,
    "video_end": 359,
    "anomaly_start": 87,
    "anomaly_end": 111,
    "anomaly_class": "ego: turning",
    "num_frames": 139,
    "subset": "val"
  },
  "UOjQsfDXjO0_001182": {
    "video_start": 1182,
    "video_end": 1295,
    "anomaly_start": 47,
    "anomaly_end": 86,
    "anomaly_class": "other: turning",
    "num_frames": 114,
    "subset": "val"
  },
  "UOjQsfDXjO0_001297": {
    "video_start": 1297,
    "video_end": 1415,
    "anomaly_start": 55,
    "anomaly_end": 80,
    "anomaly_class": "other: turning",
    "num_frames": 119,
    "subset": "val"
  },
  "UOjQsfDXjO0_001628": {
    "video_start": 1628,
    "video_end": 1716,
    "anomaly_start": 48,
    "anomaly_end": 65,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "UOjQsfDXjO0_001858": {
    "video_start": 1858,
    "video_end": 1975,
    "anomaly_start": 33,
    "anomaly_end": 62,
    "anomaly_class": "ego: turning",
    "num_frames": 118,
    "subset": "val"
  },
  "UOjQsfDXjO0_002157": {
    "video_start": 2157,
    "video_end": 2294,
    "anomaly_start": 35,
    "anomaly_end": 112,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 138,
    "subset": "val"
  },
  "UOjQsfDXjO0_002597": {
    "video_start": 2597,
    "video_end": 2754,
    "anomaly_start": 53,
    "anomaly_end": 115,
    "anomaly_class": "other: lateral",
    "num_frames": 158,
    "subset": "val"
  },
  "UOjQsfDXjO0_003077": {
    "video_start": 3077,
    "video_end": 3225,
    "anomaly_start": 85,
    "anomaly_end": 112,
    "anomaly_class": "other: oncoming",
    "num_frames": 149,
    "subset": "val"
  },
  "UOjQsfDXjO0_004953": {
    "video_start": 4953,
    "video_end": 5031,
    "anomaly_start": 21,
    "anomaly_end": 47,
    "anomaly_class": "other: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "UbDCiQVuzPY_000504": {
    "video_start": 504,
    "video_end": 602,
    "anomaly_start": 57,
    "anomaly_end": 71,
    "anomaly_class": "ego: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "UbDCiQVuzPY_000604": {
    "video_start": 604,
    "video_end": 712,
    "anomaly_start": 64,
    "anomaly_end": 86,
    "anomaly_class": "other: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "UbDCiQVuzPY_001390": {
    "video_start": 1390,
    "video_end": 1473,
    "anomaly_start": 16,
    "anomaly_end": 77,
    "anomaly_class": "other: lateral",
    "num_frames": 84,
    "subset": "val"
  },
  "UbDCiQVuzPY_001475": {
    "video_start": 1475,
    "video_end": 1566,
    "anomaly_start": 82,
    "anomaly_end": 92,
    "anomaly_class": "other: oncoming",
    "num_frames": 92,
    "subset": "val"
  },
  "UbDCiQVuzPY_002851": {
    "video_start": 2851,
    "video_end": 2949,
    "anomaly_start": 44,
    "anomaly_end": 61,
    "anomaly_class": "other: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "UbDCiQVuzPY_002951": {
    "video_start": 2951,
    "video_end": 3059,
    "anomaly_start": 40,
    "anomaly_end": 86,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 109,
    "subset": "val"
  },
  "UbDCiQVuzPY_003492": {
    "video_start": 3492,
    "video_end": 3610,
    "anomaly_start": 65,
    "anomaly_end": 84,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 119,
    "subset": "val"
  },
  "UbDCiQVuzPY_004082": {
    "video_start": 4082,
    "video_end": 4201,
    "anomaly_start": 22,
    "anomaly_end": 69,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 120,
    "subset": "val"
  },
  "UbDCiQVuzPY_004313": {
    "video_start": 4313,
    "video_end": 4451,
    "anomaly_start": 45,
    "anomaly_end": 74,
    "anomaly_class": "other: turning",
    "num_frames": 139,
    "subset": "val"
  },
  "UjkeNSr2dXQ_000394": {
    "video_start": 394,
    "video_end": 501,
    "anomaly_start": 28,
    "anomaly_end": 43,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 108,
    "subset": "val"
  },
  "UjkeNSr2dXQ_000503": {
    "video_start": 503,
    "video_end": 608,
    "anomaly_start": 42,
    "anomaly_end": 72,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 106,
    "subset": "val"
  },
  "UjkeNSr2dXQ_001217": {
    "video_start": 1217,
    "video_end": 1300,
    "anomaly_start": 47,
    "anomaly_end": 64,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 84,
    "subset": "val"
  },
  "UjkeNSr2dXQ_002046": {
    "video_start": 2046,
    "video_end": 2129,
    "anomaly_start": 49,
    "anomaly_end": 57,
    "anomaly_class": "ego: pedestrian",
    "num_frames": 84,
    "subset": "val"
  },
  "UjkeNSr2dXQ_003177": {
    "video_start": 3177,
    "video_end": 3267,
    "anomaly_start": 24,
    "anomaly_end": 36,
    "anomaly_class": "ego: turning",
    "num_frames": 91,
    "subset": "val"
  },
  "UjkeNSr2dXQ_003339": {
    "video_start": 3339,
    "video_end": 3463,
    "anomaly_start": 73,
    "anomaly_end": 85,
    "anomaly_class": "ego: oncoming",
    "num_frames": 125,
    "subset": "val"
  },
  "UjkeNSr2dXQ_003936": {
    "video_start": 3936,
    "video_end": 3992,
    "anomaly_start": 33,
    "anomaly_end": 42,
    "anomaly_class": "ego: turning",
    "num_frames": 57,
    "subset": "val"
  },
  "UjkeNSr2dXQ_004035": {
    "video_start": 4035,
    "video_end": 4133,
    "anomaly_start": 6,
    "anomaly_end": 28,
    "anomaly_class": "other: start_stop_or_stationary",
    "num_frames": 99,
    "subset": "val"
  },
  "UjkeNSr2dXQ_004437": {
    "video_start": 4437,
    "video_end": 4538,
    "anomaly_start": 35,
    "anomaly_end": 71,
    "anomaly_class": "ego: oncoming",
    "num_frames": 102,
    "subset": "val"
  },
  "UjkeNSr2dXQ_004633": {
    "video_start": 4633,
    "video_end": 4727,
    "anomaly_start": 30,
    "anomaly_end": 39,
    "anomaly_class": "ego: turning",
    "num_frames": 95,
    "subset": "val"
  },
  "V3--0ubJkNE_000150": {
    "video_start": 150,
    "video_end": 233,
    "anomaly_start": 33,
    "anomaly_end": 60,
    "anomaly_class": "ego: turning",
    "num_frames": 84,
    "subset": "val"
  },
  "V3--0ubJkNE_000874": {
    "video_start": 874,
    "video_end": 977,
    "anomaly_start": 24,
    "anomaly_end": 40,
    "anomaly_class": "ego: turning",
    "num_frames": 104,
    "subset": "val"
  },
  "V3--0ubJkNE_001382": {
    "video_start": 1382,
    "video_end": 1465,
    "anomaly_start": 40,
    "anomaly_end": 73,
    "anomaly_class": "other: turning",
    "num_frames": 84,
    "subset": "val"
  },
  "V3--0ubJkNE_001571": {
    "video_start": 1571,
    "video_end": 1658,
    "anomaly_start": 32,
    "anomaly_end": 43,
    "anomaly_class": "ego: turning",
    "num_frames": 88,
    "subset": "val"
  },
  "V3--0ubJkNE_002568": {
    "video_start": 2568,
    "video_end": 2676,
    "anomaly_start": 49,
    "anomaly_end": 70,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 109,
    "subset": "val"
  },
  "V3--0ubJkNE_002947": {
    "video_start": 2947,
    "video_end": 3045,
    "anomaly_start": 45,
    "anomaly_end": 62,
    "anomaly_class": "ego: oncoming",
    "num_frames": 99,
    "subset": "val"
  },
  "V3--0ubJkNE_003132": {
    "video_start": 3132,
    "video_end": 3219,
    "anomaly_start": 32,
    "anomaly_end": 37,
    "anomaly_class": "ego: turning",
    "num_frames": 88,
    "subset": "val"
  },
  "V3--0ubJkNE_003793": {
    "video_start": 3793,
    "video_end": 3876,
    "anomaly_start": 32,
    "anomaly_end": 43,
    "anomaly_class": "ego: lateral",
    "num_frames": 84,
    "subset": "val"
  },
  "V3--0ubJkNE_004309": {
    "video_start": 4309,
    "video_end": 4392,
    "anomaly_start": 27,
    "anomaly_end": 45,
    "anomaly_class": "ego: turning",
    "num_frames": 84,
    "subset": "val"
  },
  "V3--0ubJkNE_005532": {
    "video_start": 5532,
    "video_end": 5609,
    "anomaly_start": 32,
    "anomaly_end": 57,
    "anomaly_class": "ego: turning",
    "num_frames": 78,
    "subset": "val"
  },
  "V3--0ubJkNE_006107": {
    "video_start": 6107,
    "video_end": 6215,
    "anomaly_start": 39,
    "anomaly_end": 68,
    "anomaly_class": "other: lateral",
    "num_frames": 109,
    "subset": "val"
  },
  "V3--0ubJkNE_006217": {
    "video_start": 6217,
    "video_end": 6374,
    "anomaly_start": 88,
    "anomaly_end": 143,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 158,
    "subset": "val"
  },
  "Vtpe3HAIEm4_000194": {
    "video_start": 194,
    "video_end": 300,
    "anomaly_start": 8,
    "anomaly_end": 27,
    "anomaly_class": "other: lateral",
    "num_frames": 107,
    "subset": "val"
  },
  "Vtpe3HAIEm4_000631": {
    "video_start": 631,
    "video_end": 727,
    "anomaly_start": 30,
    "anomaly_end": 51,
    "anomaly_class": "ego: lateral",
    "num_frames": 97,
    "subset": "val"
  },
  "Vtpe3HAIEm4_002165": {
    "video_start": 2165,
    "video_end": 2240,
    "anomaly_start": 36,
    "anomaly_end": 52,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 76,
    "subset": "val"
  },
  "Vtpe3HAIEm4_002565": {
    "video_start": 2565,
    "video_end": 2654,
    "anomaly_start": 24,
    "anomaly_end": 46,
    "anomaly_class": "other: turning",
    "num_frames": 90,
    "subset": "val"
  },
  "Vtpe3HAIEm4_004006": {
    "video_start": 4006,
    "video_end": 4086,
    "anomaly_start": 36,
    "anomaly_end": 67,
    "anomaly_class": "ego: lateral",
    "num_frames": 81,
    "subset": "val"
  },
  "Vtpe3HAIEm4_005234": {
    "video_start": 5234,
    "video_end": 5302,
    "anomaly_start": 23,
    "anomaly_end": 47,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 69,
    "subset": "val"
  },
  "Vtpe3HAIEm4_005304": {
    "video_start": 5304,
    "video_end": 5372,
    "anomaly_start": 22,
    "anomaly_end": 47,
    "anomaly_class": "other: lateral",
    "num_frames": 69,
    "subset": "val"
  },
  "Vtpe3HAIEm4_005624": {
    "video_start": 5624,
    "video_end": 5744,
    "anomaly_start": 22,
    "anomaly_end": 55,
    "anomaly_class": "other: turning",
    "num_frames": 121,
    "subset": "val"
  },
  "VxWJchENocA_000308": {
    "video_start": 308,
    "video_end": 398,
    "anomaly_start": 42,
    "anomaly_end": 71,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 91,
    "subset": "val"
  },
  "VxWJchENocA_000566": {
    "video_start": 566,
    "video_end": 656,
    "anomaly_start": 35,
    "anomaly_end": 69,
    "anomaly_class": "other: turning",
    "num_frames": 91,
    "subset": "val"
  },
  "VxWJchENocA_000658": {
    "video_start": 658,
    "video_end": 728,
    "anomaly_start": 33,
    "anomaly_end": 52,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 71,
    "subset": "val"
  },
  "VxWJchENocA_001660": {
    "video_start": 1660,
    "video_end": 1732,
    "anomaly_start": 20,
    "anomaly_end": 46,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 73,
    "subset": "val"
  },
  "VxWJchENocA_001734": {
    "video_start": 1734,
    "video_end": 1825,
    "anomaly_start": 42,
    "anomaly_end": 68,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 92,
    "subset": "val"
  },
  "VxWJchENocA_001910": {
    "video_start": 1910,
    "video_end": 2021,
    "anomaly_start": 26,
    "anomaly_end": 59,
    "anomaly_class": "ego: lateral",
    "num_frames": 112,
    "subset": "val"
  },
  "VxWJchENocA_002834": {
    "video_start": 2834,
    "video_end": 2926,
    "anomaly_start": 15,
    "anomaly_end": 72,
    "anomaly_class": "ego: lateral",
    "num_frames": 93,
    "subset": "val"
  },
  "VxWJchENocA_003032": {
    "video_start": 3032,
    "video_end": 3101,
    "anomaly_start": 20,
    "anomaly_end": 46,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 70,
    "subset": "val"
  },
  "VxWJchENocA_003254": {
    "video_start": 3254,
    "video_end": 3340,
    "anomaly_start": 44,
    "anomaly_end": 62,
    "anomaly_class": "ego: oncoming",
    "num_frames": 87,
    "subset": "val"
  },
  "VxWJchENocA_003610": {
    "video_start": 3610,
    "video_end": 3654,
    "anomaly_start": 28,
    "anomaly_end": 45,
    "anomaly_class": "ego: pedestrian",
    "num_frames": 45,
    "subset": "val"
  },
  "VxWJchENocA_004081": {
    "video_start": 4081,
    "video_end": 4196,
    "anomaly_start": 39,
    "anomaly_end": 63,
    "anomaly_class": "ego: lateral",
    "num_frames": 116,
    "subset": "val"
  },
  "VxWJchENocA_004338": {
    "video_start": 4338,
    "video_end": 4450,
    "anomaly_start": 81,
    "anomaly_end": 97,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 113,
    "subset": "val"
  },
  "W6YrlYyWguc_000198": {
    "video_start": 198,
    "video_end": 268,
    "anomaly_start": 65,
    "anomaly_end": 71,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 71,
    "subset": "val"
  },
  "W6YrlYyWguc_000856": {
    "video_start": 856,
    "video_end": 994,
    "anomaly_start": 51,
    "anomaly_end": 77,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 139,
    "subset": "val"
  },
  "W6YrlYyWguc_002556": {
    "video_start": 2556,
    "video_end": 2664,
    "anomaly_start": 40,
    "anomaly_end": 82,
    "anomaly_class": "other: lateral",
    "num_frames": 109,
    "subset": "val"
  },
  "W6YrlYyWguc_002776": {
    "video_start": 2776,
    "video_end": 2875,
    "anomaly_start": 59,
    "anomaly_end": 73,
    "anomaly_class": "ego: unknown",
    "num_frames": 100,
    "subset": "val"
  },
  "W6YrlYyWguc_003216": {
    "video_start": 3216,
    "video_end": 3324,
    "anomaly_start": 42,
    "anomaly_end": 62,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 109,
    "subset": "val"
  },
  "W6YrlYyWguc_003326": {
    "video_start": 3326,
    "video_end": 3444,
    "anomaly_start": 39,
    "anomaly_end": 87,
    "anomaly_class": "other: lateral",
    "num_frames": 119,
    "subset": "val"
  },
  "W6YrlYyWguc_003446": {
    "video_start": 3446,
    "video_end": 3544,
    "anomaly_start": 33,
    "anomaly_end": 63,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 99,
    "subset": "val"
  },
  "W6YrlYyWguc_003546": {
    "video_start": 3546,
    "video_end": 3634,
    "anomaly_start": 48,
    "anomaly_end": 61,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "W6YrlYyWguc_003816": {
    "video_start": 3816,
    "video_end": 3895,
    "anomaly_start": 60,
    "anomaly_end": 75,
    "anomaly_class": "ego: turning",
    "num_frames": 80,
    "subset": "val"
  },
  "W6YrlYyWguc_003897": {
    "video_start": 3897,
    "video_end": 4015,
    "anomaly_start": 71,
    "anomaly_end": 96,
    "anomaly_class": "other: turning",
    "num_frames": 119,
    "subset": "val"
  },
  "W6YrlYyWguc_005116": {
    "video_start": 5116,
    "video_end": 5265,
    "anomaly_start": 87,
    "anomaly_end": 137,
    "anomaly_class": "other: start_stop_or_stationary",
    "num_frames": 150,
    "subset": "val"
  },
  "W6YrlYyWguc_005367": {
    "video_start": 5367,
    "video_end": 5475,
    "anomaly_start": 42,
    "anomaly_end": 84,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 109,
    "subset": "val"
  },
  "W6YrlYyWguc_005597": {
    "video_start": 5597,
    "video_end": 5685,
    "anomaly_start": 41,
    "anomaly_end": 58,
    "anomaly_class": "ego: unknown",
    "num_frames": 89,
    "subset": "val"
  },
  "W6YrlYyWguc_005927": {
    "video_start": 5927,
    "video_end": 6021,
    "anomaly_start": 55,
    "anomaly_end": 64,
    "anomaly_class": "other: turning",
    "num_frames": 95,
    "subset": "val"
  },
  "WWyeuvpHmfc_000456": {
    "video_start": 456,
    "video_end": 545,
    "anomaly_start": 41,
    "anomaly_end": 61,
    "anomaly_class": "ego: obstacle",
    "num_frames": 90,
    "subset": "val"
  },
  "WWyeuvpHmfc_000835": {
    "video_start": 835,
    "video_end": 935,
    "anomaly_start": 8,
    "anomaly_end": 30,
    "anomaly_class": "ego: lateral",
    "num_frames": 101,
    "subset": "val"
  },
  "WWyeuvpHmfc_001275": {
    "video_start": 1275,
    "video_end": 1361,
    "anomaly_start": 34,
    "anomaly_end": 55,
    "anomaly_class": "ego: lateral",
    "num_frames": 87,
    "subset": "val"
  },
  "WWyeuvpHmfc_002174": {
    "video_start": 2174,
    "video_end": 2249,
    "anomaly_start": 17,
    "anomaly_end": 37,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 76,
    "subset": "val"
  },
  "WWyeuvpHmfc_002770": {
    "video_start": 2770,
    "video_end": 2846,
    "anomaly_start": 28,
    "anomaly_end": 53,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 77,
    "subset": "val"
  },
  "WWyeuvpHmfc_003931": {
    "video_start": 3931,
    "video_end": 4008,
    "anomaly_start": 15,
    "anomaly_end": 33,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 78,
    "subset": "val"
  },
  "WWyeuvpHmfc_004101": {
    "video_start": 4101,
    "video_end": 4131,
    "anomaly_start": 21,
    "anomaly_end": 31,
    "anomaly_class": "ego: turning",
    "num_frames": 31,
    "subset": "val"
  },
  "WWyeuvpHmfc_004370": {
    "video_start": 4370,
    "video_end": 4467,
    "anomaly_start": 35,
    "anomaly_end": 57,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 98,
    "subset": "val"
  },
  "WWyeuvpHmfc_004796": {
    "video_start": 4796,
    "video_end": 4878,
    "anomaly_start": 50,
    "anomaly_end": 58,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 83,
    "subset": "val"
  },
  "X8VgGb1fJDU_000388": {
    "video_start": 388,
    "video_end": 466,
    "anomaly_start": 36,
    "anomaly_end": 63,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 79,
    "subset": "val"
  },
  "X8VgGb1fJDU_000468": {
    "video_start": 468,
    "video_end": 551,
    "anomaly_start": 29,
    "anomaly_end": 84,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 84,
    "subset": "val"
  },
  "X8VgGb1fJDU_001122": {
    "video_start": 1122,
    "video_end": 1179,
    "anomaly_start": 7,
    "anomaly_end": 24,
    "anomaly_class": "other: turning",
    "num_frames": 58,
    "subset": "val"
  },
  "X8VgGb1fJDU_003014": {
    "video_start": 3014,
    "video_end": 3090,
    "anomaly_start": 32,
    "anomaly_end": 57,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 77,
    "subset": "val"
  },
  "Y4rb2_HaBdQ_000052": {
    "video_start": 52,
    "video_end": 130,
    "anomaly_start": 35,
    "anomaly_end": 55,
    "anomaly_class": "ego: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "Y4rb2_HaBdQ_000329": {
    "video_start": 329,
    "video_end": 414,
    "anomaly_start": 20,
    "anomaly_end": 47,
    "anomaly_class": "ego: turning",
    "num_frames": 86,
    "subset": "val"
  },
  "Y4rb2_HaBdQ_000879": {
    "video_start": 879,
    "video_end": 996,
    "anomaly_start": 35,
    "anomaly_end": 56,
    "anomaly_class": "ego: lateral",
    "num_frames": 118,
    "subset": "val"
  },
  "Y4rb2_HaBdQ_001268": {
    "video_start": 1268,
    "video_end": 1360,
    "anomaly_start": 49,
    "anomaly_end": 81,
    "anomaly_class": "ego: turning",
    "num_frames": 93,
    "subset": "val"
  },
  "Y4rb2_HaBdQ_001466": {
    "video_start": 1466,
    "video_end": 1569,
    "anomaly_start": 33,
    "anomaly_end": 50,
    "anomaly_class": "ego: obstacle",
    "num_frames": 104,
    "subset": "val"
  },
  "Y4rb2_HaBdQ_001716": {
    "video_start": 1716,
    "video_end": 1777,
    "anomaly_start": 45,
    "anomaly_end": 62,
    "anomaly_class": "ego: turning",
    "num_frames": 62,
    "subset": "val"
  },
  "Y4rb2_HaBdQ_002441": {
    "video_start": 2441,
    "video_end": 2526,
    "anomaly_start": 42,
    "anomaly_end": 67,
    "anomaly_class": "ego: turning",
    "num_frames": 86,
    "subset": "val"
  },
  "Y4rb2_HaBdQ_002928": {
    "video_start": 2928,
    "video_end": 3053,
    "anomaly_start": 38,
    "anomaly_end": 56,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 126,
    "subset": "val"
  },
  "Y4rb2_HaBdQ_003578": {
    "video_start": 3578,
    "video_end": 3655,
    "anomaly_start": 11,
    "anomaly_end": 42,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 78,
    "subset": "val"
  },
  "Y4rb2_HaBdQ_003915": {
    "video_start": 3915,
    "video_end": 4020,
    "anomaly_start": 48,
    "anomaly_end": 66,
    "anomaly_class": "ego: turning",
    "num_frames": 106,
    "subset": "val"
  },
  "Y4rb2_HaBdQ_004472": {
    "video_start": 4472,
    "video_end": 4591,
    "anomaly_start": 87,
    "anomaly_end": 108,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 120,
    "subset": "val"
  },
  "Y4rb2_HaBdQ_005046": {
    "video_start": 5046,
    "video_end": 5186,
    "anomaly_start": 35,
    "anomaly_end": 61,
    "anomaly_class": "other: turning",
    "num_frames": 141,
    "subset": "val"
  },
  "YBEYOS3A3Ic_001426": {
    "video_start": 1426,
    "video_end": 1524,
    "anomaly_start": 39,
    "anomaly_end": 67,
    "anomaly_class": "ego: lateral",
    "num_frames": 99,
    "subset": "val"
  },
  "YBEYOS3A3Ic_002298": {
    "video_start": 2298,
    "video_end": 2386,
    "anomaly_start": 46,
    "anomaly_end": 64,
    "anomaly_class": "other: lateral",
    "num_frames": 89,
    "subset": "val"
  },
  "YBEYOS3A3Ic_003122": {
    "video_start": 3122,
    "video_end": 3213,
    "anomaly_start": 12,
    "anomaly_end": 35,
    "anomaly_class": "other: turning",
    "num_frames": 92,
    "subset": "val"
  },
  "YBEYOS3A3Ic_004305": {
    "video_start": 4305,
    "video_end": 4383,
    "anomaly_start": 22,
    "anomaly_end": 43,
    "anomaly_class": "ego: lateral",
    "num_frames": 79,
    "subset": "val"
  },
  "YBEYOS3A3Ic_004577": {
    "video_start": 4577,
    "video_end": 4654,
    "anomaly_start": 37,
    "anomaly_end": 71,
    "anomaly_class": "ego: turning",
    "num_frames": 78,
    "subset": "val"
  },
  "YBEYOS3A3Ic_005264": {
    "video_start": 5264,
    "video_end": 5347,
    "anomaly_start": 48,
    "anomaly_end": 66,
    "anomaly_class": "other: turning",
    "num_frames": 84,
    "subset": "val"
  },
  "YBEYOS3A3Ic_005749": {
    "video_start": 5749,
    "video_end": 5828,
    "anomaly_start": 22,
    "anomaly_end": 50,
    "anomaly_class": "ego: turning",
    "num_frames": 80,
    "subset": "val"
  },
  "Z9K13eBUwJM_000135": {
    "video_start": 135,
    "video_end": 168,
    "anomaly_start": 18,
    "anomaly_end": 34,
    "anomaly_class": "ego: oncoming",
    "num_frames": 34,
    "subset": "val"
  },
  "Z9K13eBUwJM_001051": {
    "video_start": 1051,
    "video_end": 1165,
    "anomaly_start": 31,
    "anomaly_end": 100,
    "anomaly_class": "ego: lateral",
    "num_frames": 115,
    "subset": "val"
  },
  "Z9K13eBUwJM_003427": {
    "video_start": 3427,
    "video_end": 3516,
    "anomaly_start": 54,
    "anomaly_end": 66,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 90,
    "subset": "val"
  },
  "Z9K13eBUwJM_004677": {
    "video_start": 4677,
    "video_end": 4743,
    "anomaly_start": 23,
    "anomaly_end": 29,
    "anomaly_class": "ego: oncoming",
    "num_frames": 67,
    "subset": "val"
  },
  "Z9K13eBUwJM_005281": {
    "video_start": 5281,
    "video_end": 5374,
    "anomaly_start": 40,
    "anomaly_end": 51,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 94,
    "subset": "val"
  },
  "Z9K13eBUwJM_006190": {
    "video_start": 6190,
    "video_end": 6274,
    "anomaly_start": 31,
    "anomaly_end": 81,
    "anomaly_class": "ego: oncoming",
    "num_frames": 85,
    "subset": "val"
  },
  "ZIts2XH28SA_000075": {
    "video_start": 75,
    "video_end": 169,
    "anomaly_start": 22,
    "anomaly_end": 47,
    "anomaly_class": "ego: oncoming",
    "num_frames": 95,
    "subset": "val"
  },
  "ZIts2XH28SA_001272": {
    "video_start": 1272,
    "video_end": 1392,
    "anomaly_start": 67,
    "anomaly_end": 121,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 121,
    "subset": "val"
  },
  "ZIts2XH28SA_001782": {
    "video_start": 1782,
    "video_end": 1880,
    "anomaly_start": 38,
    "anomaly_end": 63,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 99,
    "subset": "val"
  },
  "ZIts2XH28SA_002402": {
    "video_start": 2402,
    "video_end": 2500,
    "anomaly_start": 50,
    "anomaly_end": 69,
    "anomaly_class": "other: lateral",
    "num_frames": 99,
    "subset": "val"
  },
  "ZIts2XH28SA_003134": {
    "video_start": 3134,
    "video_end": 3259,
    "anomaly_start": 38,
    "anomaly_end": 126,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 126,
    "subset": "val"
  },
  "ZIts2XH28SA_003537": {
    "video_start": 3537,
    "video_end": 3655,
    "anomaly_start": 25,
    "anomaly_end": 74,
    "anomaly_class": "ego: lateral",
    "num_frames": 119,
    "subset": "val"
  },
  "ZIts2XH28SA_003657": {
    "video_start": 3657,
    "video_end": 3751,
    "anomaly_start": 20,
    "anomaly_end": 36,
    "anomaly_class": "ego: unknown",
    "num_frames": 95,
    "subset": "val"
  },
  "ZIts2XH28SA_003848": {
    "video_start": 3848,
    "video_end": 3962,
    "anomaly_start": 43,
    "anomaly_end": 73,
    "anomaly_class": "ego: oncoming",
    "num_frames": 115,
    "subset": "val"
  },
  "ZIts2XH28SA_004402": {
    "video_start": 4402,
    "video_end": 4520,
    "anomaly_start": 48,
    "anomaly_end": 76,
    "anomaly_class": "ego: unknown",
    "num_frames": 119,
    "subset": "val"
  },
  "Zpo0kwg1XEQ_000321": {
    "video_start": 321,
    "video_end": 408,
    "anomaly_start": 37,
    "anomaly_end": 63,
    "anomaly_class": "ego: lateral",
    "num_frames": 88,
    "subset": "val"
  },
  "Zpo0kwg1XEQ_000720": {
    "video_start": 720,
    "video_end": 809,
    "anomaly_start": 41,
    "anomaly_end": 54,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 90,
    "subset": "val"
  },
  "Zpo0kwg1XEQ_001043": {
    "video_start": 1043,
    "video_end": 1132,
    "anomaly_start": 32,
    "anomaly_end": 54,
    "anomaly_class": "other: turning",
    "num_frames": 90,
    "subset": "val"
  },
  "Zpo0kwg1XEQ_002029": {
    "video_start": 2029,
    "video_end": 2129,
    "anomaly_start": 25,
    "anomaly_end": 44,
    "anomaly_class": "other: turning",
    "num_frames": 101,
    "subset": "val"
  },
  "Zpo0kwg1XEQ_002482": {
    "video_start": 2482,
    "video_end": 2578,
    "anomaly_start": 28,
    "anomaly_end": 63,
    "anomaly_class": "ego: oncoming",
    "num_frames": 97,
    "subset": "val"
  },
  "Zpo0kwg1XEQ_003224": {
    "video_start": 3224,
    "video_end": 3320,
    "anomaly_start": 54,
    "anomaly_end": 83,
    "anomaly_class": "ego: lateral",
    "num_frames": 97,
    "subset": "val"
  },
  "Zpo0kwg1XEQ_003952": {
    "video_start": 3952,
    "video_end": 4030,
    "anomaly_start": 37,
    "anomaly_end": 61,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 79,
    "subset": "val"
  },
  "Zpo0kwg1XEQ_004129": {
    "video_start": 4129,
    "video_end": 4195,
    "anomaly_start": 29,
    "anomaly_end": 44,
    "anomaly_class": "ego: turning",
    "num_frames": 67,
    "subset": "val"
  },
  "Zpo0kwg1XEQ_004447": {
    "video_start": 4447,
    "video_end": 4510,
    "anomaly_start": 16,
    "anomaly_end": 43,
    "anomaly_class": "other: turning",
    "num_frames": 64,
    "subset": "val"
  },
  "a_VxrUq9PmA_000072": {
    "video_start": 72,
    "video_end": 179,
    "anomaly_start": 47,
    "anomaly_end": 62,
    "anomaly_class": "ego: turning",
    "num_frames": 108,
    "subset": "val"
  },
  "a_VxrUq9PmA_000181": {
    "video_start": 181,
    "video_end": 384,
    "anomaly_start": 53,
    "anomaly_end": 64,
    "anomaly_class": "ego: oncoming",
    "num_frames": 204,
    "subset": "val"
  },
  "a_VxrUq9PmA_000656": {
    "video_start": 656,
    "video_end": 754,
    "anomaly_start": 26,
    "anomaly_end": 79,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 99,
    "subset": "val"
  },
  "a_VxrUq9PmA_000867": {
    "video_start": 867,
    "video_end": 975,
    "anomaly_start": 62,
    "anomaly_end": 86,
    "anomaly_class": "other: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "a_VxrUq9PmA_000977": {
    "video_start": 977,
    "video_end": 1135,
    "anomaly_start": 61,
    "anomaly_end": 108,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 159,
    "subset": "val"
  },
  "a_VxrUq9PmA_001137": {
    "video_start": 1137,
    "video_end": 1245,
    "anomaly_start": 40,
    "anomaly_end": 96,
    "anomaly_class": "other: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "a_VxrUq9PmA_001508": {
    "video_start": 1508,
    "video_end": 1606,
    "anomaly_start": 36,
    "anomaly_end": 52,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 99,
    "subset": "val"
  },
  "a_VxrUq9PmA_001608": {
    "video_start": 1608,
    "video_end": 1716,
    "anomaly_start": 35,
    "anomaly_end": 108,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 109,
    "subset": "val"
  },
  "a_VxrUq9PmA_001957": {
    "video_start": 1957,
    "video_end": 2095,
    "anomaly_start": 26,
    "anomaly_end": 80,
    "anomaly_class": "other: oncoming",
    "num_frames": 139,
    "subset": "val"
  },
  "a_VxrUq9PmA_002357": {
    "video_start": 2357,
    "video_end": 2495,
    "anomaly_start": 61,
    "anomaly_end": 84,
    "anomaly_class": "other: turning",
    "num_frames": 139,
    "subset": "val"
  },
  "a_VxrUq9PmA_002682": {
    "video_start": 2682,
    "video_end": 2785,
    "anomaly_start": 24,
    "anomaly_end": 75,
    "anomaly_class": "other: lateral",
    "num_frames": 104,
    "subset": "val"
  },
  "a_VxrUq9PmA_003832": {
    "video_start": 3832,
    "video_end": 3971,
    "anomaly_start": 36,
    "anomaly_end": 85,
    "anomaly_class": "other: turning",
    "num_frames": 140,
    "subset": "val"
  },
  "ahKX0rtdMJc_000073": {
    "video_start": 73,
    "video_end": 179,
    "anomaly_start": 44,
    "anomaly_end": 68,
    "anomaly_class": "other: turning",
    "num_frames": 107,
    "subset": "val"
  },
  "ahKX0rtdMJc_000473": {
    "video_start": 473,
    "video_end": 670,
    "anomaly_start": 17,
    "anomaly_end": 51,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 198,
    "subset": "val"
  },
  "ahKX0rtdMJc_000962": {
    "video_start": 962,
    "video_end": 1130,
    "anomaly_start": 59,
    "anomaly_end": 134,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 169,
    "subset": "val"
  },
  "ahKX0rtdMJc_001382": {
    "video_start": 1382,
    "video_end": 1480,
    "anomaly_start": 57,
    "anomaly_end": 71,
    "anomaly_class": "ego: oncoming",
    "num_frames": 99,
    "subset": "val"
  },
  "ahKX0rtdMJc_001482": {
    "video_start": 1482,
    "video_end": 1581,
    "anomaly_start": 55,
    "anomaly_end": 74,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 100,
    "subset": "val"
  },
  "ahKX0rtdMJc_002138": {
    "video_start": 2138,
    "video_end": 2226,
    "anomaly_start": 32,
    "anomaly_end": 70,
    "anomaly_class": "ego: oncoming",
    "num_frames": 89,
    "subset": "val"
  },
  "ahKX0rtdMJc_002428": {
    "video_start": 2428,
    "video_end": 2526,
    "anomaly_start": 49,
    "anomaly_end": 65,
    "anomaly_class": "ego: oncoming",
    "num_frames": 99,
    "subset": "val"
  },
  "ahKX0rtdMJc_002528": {
    "video_start": 2528,
    "video_end": 2639,
    "anomaly_start": 43,
    "anomaly_end": 60,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 112,
    "subset": "val"
  },
  "ahKX0rtdMJc_002641": {
    "video_start": 2641,
    "video_end": 2749,
    "anomaly_start": 36,
    "anomaly_end": 89,
    "anomaly_class": "other: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "ahKX0rtdMJc_002851": {
    "video_start": 2851,
    "video_end": 2949,
    "anomaly_start": 65,
    "anomaly_end": 99,
    "anomaly_class": "ego: unknown",
    "num_frames": 99,
    "subset": "val"
  },
  "ahKX0rtdMJc_003639": {
    "video_start": 3639,
    "video_end": 3737,
    "anomaly_start": 22,
    "anomaly_end": 99,
    "anomaly_class": "other: unknown",
    "num_frames": 99,
    "subset": "val"
  },
  "ahKX0rtdMJc_004382": {
    "video_start": 4382,
    "video_end": 4477,
    "anomaly_start": 44,
    "anomaly_end": 88,
    "anomaly_class": "other: turning",
    "num_frames": 96,
    "subset": "val"
  },
  "ahKX0rtdMJc_005345": {
    "video_start": 5345,
    "video_end": 5423,
    "anomaly_start": 28,
    "anomaly_end": 48,
    "anomaly_class": "other: lateral",
    "num_frames": 79,
    "subset": "val"
  },
  "bFGmOp9H3MA_000355": {
    "video_start": 355,
    "video_end": 592,
    "anomaly_start": 35,
    "anomaly_end": 238,
    "anomaly_class": "ego: oncoming",
    "num_frames": 238,
    "subset": "val"
  },
  "bFGmOp9H3MA_000594": {
    "video_start": 594,
    "video_end": 684,
    "anomaly_start": 17,
    "anomaly_end": 75,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 91,
    "subset": "val"
  },
  "bFGmOp9H3MA_000853": {
    "video_start": 853,
    "video_end": 937,
    "anomaly_start": 49,
    "anomaly_end": 82,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 85,
    "subset": "val"
  },
  "bFGmOp9H3MA_001832": {
    "video_start": 1832,
    "video_end": 1985,
    "anomaly_start": 36,
    "anomaly_end": 154,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 154,
    "subset": "val"
  },
  "bFGmOp9H3MA_001987": {
    "video_start": 1987,
    "video_end": 2087,
    "anomaly_start": 41,
    "anomaly_end": 90,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 101,
    "subset": "val"
  },
  "bFGmOp9H3MA_002646": {
    "video_start": 2646,
    "video_end": 2754,
    "anomaly_start": 35,
    "anomaly_end": 66,
    "anomaly_class": "ego: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "bFGmOp9H3MA_002953": {
    "video_start": 2953,
    "video_end": 3040,
    "anomaly_start": 19,
    "anomaly_end": 40,
    "anomaly_class": "ego: turning",
    "num_frames": 88,
    "subset": "val"
  },
  "bFGmOp9H3MA_003113": {
    "video_start": 3113,
    "video_end": 3206,
    "anomaly_start": 33,
    "anomaly_end": 59,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 94,
    "subset": "val"
  },
  "bFGmOp9H3MA_003708": {
    "video_start": 3708,
    "video_end": 3832,
    "anomaly_start": 35,
    "anomaly_end": 116,
    "anomaly_class": "ego: turning",
    "num_frames": 125,
    "subset": "val"
  },
  "bFGmOp9H3MA_005278": {
    "video_start": 5278,
    "video_end": 5401,
    "anomaly_start": 33,
    "anomaly_end": 61,
    "anomaly_class": "ego: lateral",
    "num_frames": 124,
    "subset": "val"
  },
  "bFGmOp9H3MA_005756": {
    "video_start": 5756,
    "video_end": 5888,
    "anomaly_start": 23,
    "anomaly_end": 111,
    "anomaly_class": "other: unknown",
    "num_frames": 133,
    "subset": "val"
  },
  "bI9JzQSrEGo_000680": {
    "video_start": 680,
    "video_end": 777,
    "anomaly_start": 38,
    "anomaly_end": 74,
    "anomaly_class": "ego: oncoming",
    "num_frames": 98,
    "subset": "val"
  },
  "bI9JzQSrEGo_001743": {
    "video_start": 1743,
    "video_end": 1812,
    "anomaly_start": 27,
    "anomaly_end": 69,
    "anomaly_class": "ego: turning",
    "num_frames": 70,
    "subset": "val"
  },
  "bI9JzQSrEGo_003380": {
    "video_start": 3380,
    "video_end": 3427,
    "anomaly_start": 34,
    "anomaly_end": 48,
    "anomaly_class": "ego: lateral",
    "num_frames": 48,
    "subset": "val"
  },
  "bI9JzQSrEGo_003840": {
    "video_start": 3840,
    "video_end": 3942,
    "anomaly_start": 44,
    "anomaly_end": 71,
    "anomaly_class": "ego: turning",
    "num_frames": 103,
    "subset": "val"
  },
  "bI9JzQSrEGo_004441": {
    "video_start": 4441,
    "video_end": 4532,
    "anomaly_start": 31,
    "anomaly_end": 64,
    "anomaly_class": "ego: turning",
    "num_frames": 92,
    "subset": "val"
  },
  "bI9JzQSrEGo_004672": {
    "video_start": 4672,
    "video_end": 4829,
    "anomaly_start": 60,
    "anomaly_end": 136,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 158,
    "subset": "val"
  },
  "bI9JzQSrEGo_004831": {
    "video_start": 4831,
    "video_end": 4899,
    "anomaly_start": 44,
    "anomaly_end": 69,
    "anomaly_class": "ego: turning",
    "num_frames": 69,
    "subset": "val"
  },
  "bI9JzQSrEGo_004901": {
    "video_start": 4901,
    "video_end": 4987,
    "anomaly_start": 53,
    "anomaly_end": 84,
    "anomaly_class": "ego: oncoming",
    "num_frames": 87,
    "subset": "val"
  },
  "bcpfYpmDcp0_000541": {
    "video_start": 541,
    "video_end": 622,
    "anomaly_start": 25,
    "anomaly_end": 66,
    "anomaly_class": "ego: lateral",
    "num_frames": 82,
    "subset": "val"
  },
  "bcpfYpmDcp0_000972": {
    "video_start": 972,
    "video_end": 1130,
    "anomaly_start": 30,
    "anomaly_end": 31,
    "anomaly_class": "other: unknown",
    "num_frames": 159,
    "subset": "val"
  },
  "bcpfYpmDcp0_002246": {
    "video_start": 2246,
    "video_end": 2338,
    "anomaly_start": 40,
    "anomaly_end": 71,
    "anomaly_class": "ego: lateral",
    "num_frames": 93,
    "subset": "val"
  },
  "bcpfYpmDcp0_003790": {
    "video_start": 3790,
    "video_end": 3879,
    "anomaly_start": 46,
    "anomaly_end": 90,
    "anomaly_class": "ego: oncoming",
    "num_frames": 90,
    "subset": "val"
  },
  "bcpfYpmDcp0_003881": {
    "video_start": 3881,
    "video_end": 3952,
    "anomaly_start": 19,
    "anomaly_end": 35,
    "anomaly_class": "other: obstacle",
    "num_frames": 72,
    "subset": "val"
  },
  "bcpfYpmDcp0_004277": {
    "video_start": 4277,
    "video_end": 4348,
    "anomaly_start": 28,
    "anomaly_end": 57,
    "anomaly_class": "ego: lateral",
    "num_frames": 72,
    "subset": "val"
  },
  "bhA2ckvE-TQ_000072": {
    "video_start": 72,
    "video_end": 167,
    "anomaly_start": 64,
    "anomaly_end": 96,
    "anomaly_class": "ego: turning",
    "num_frames": 96,
    "subset": "val"
  },
  "bhA2ckvE-TQ_000174": {
    "video_start": 174,
    "video_end": 309,
    "anomaly_start": 31,
    "anomaly_end": 65,
    "anomaly_class": "other: turning",
    "num_frames": 136,
    "subset": "val"
  },
  "bhA2ckvE-TQ_000311": {
    "video_start": 311,
    "video_end": 419,
    "anomaly_start": 55,
    "anomaly_end": 87,
    "anomaly_class": "other: oncoming",
    "num_frames": 109,
    "subset": "val"
  },
  "bhA2ckvE-TQ_000722": {
    "video_start": 722,
    "video_end": 834,
    "anomaly_start": 33,
    "anomaly_end": 104,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 113,
    "subset": "val"
  },
  "bhA2ckvE-TQ_001691": {
    "video_start": 1691,
    "video_end": 1795,
    "anomaly_start": 25,
    "anomaly_end": 71,
    "anomaly_class": "ego: lateral",
    "num_frames": 105,
    "subset": "val"
  },
  "bhA2ckvE-TQ_002182": {
    "video_start": 2182,
    "video_end": 2320,
    "anomaly_start": 33,
    "anomaly_end": 124,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 139,
    "subset": "val"
  },
  "bhA2ckvE-TQ_002807": {
    "video_start": 2807,
    "video_end": 2905,
    "anomaly_start": 40,
    "anomaly_end": 99,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 99,
    "subset": "val"
  },
  "bhA2ckvE-TQ_003227": {
    "video_start": 3227,
    "video_end": 3399,
    "anomaly_start": 49,
    "anomaly_end": 96,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 173,
    "subset": "val"
  },
  "bhA2ckvE-TQ_003401": {
    "video_start": 3401,
    "video_end": 3510,
    "anomaly_start": 29,
    "anomaly_end": 64,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 110,
    "subset": "val"
  },
  "bhA2ckvE-TQ_004401": {
    "video_start": 4401,
    "video_end": 4498,
    "anomaly_start": 45,
    "anomaly_end": 97,
    "anomaly_class": "ego: pedestrian",
    "num_frames": 98,
    "subset": "val"
  },
  "bhA2ckvE-TQ_004788": {
    "video_start": 4788,
    "video_end": 4956,
    "anomaly_start": 21,
    "anomaly_end": 48,
    "anomaly_class": "other: turning",
    "num_frames": 169,
    "subset": "val"
  },
  "bhA2ckvE-TQ_005336": {
    "video_start": 5336,
    "video_end": 5424,
    "anomaly_start": 42,
    "anomaly_end": 66,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 89,
    "subset": "val"
  },
  "bhA2ckvE-TQ_005426": {
    "video_start": 5426,
    "video_end": 5538,
    "anomaly_start": 23,
    "anomaly_end": 83,
    "anomaly_class": "other: unknown",
    "num_frames": 113,
    "subset": "val"
  },
  "cfrLchAShxQ_000212": {
    "video_start": 212,
    "video_end": 308,
    "anomaly_start": 27,
    "anomaly_end": 78,
    "anomaly_class": "other: turning",
    "num_frames": 97,
    "subset": "val"
  },
  "cfrLchAShxQ_000485": {
    "video_start": 485,
    "video_end": 600,
    "anomaly_start": 24,
    "anomaly_end": 89,
    "anomaly_class": "ego: turning",
    "num_frames": 116,
    "subset": "val"
  },
  "cfrLchAShxQ_000602": {
    "video_start": 602,
    "video_end": 750,
    "anomaly_start": 49,
    "anomaly_end": 103,
    "anomaly_class": "other: lateral",
    "num_frames": 149,
    "subset": "val"
  },
  "cfrLchAShxQ_001515": {
    "video_start": 1515,
    "video_end": 1584,
    "anomaly_start": 19,
    "anomaly_end": 70,
    "anomaly_class": "ego: lateral",
    "num_frames": 70,
    "subset": "val"
  },
  "cfrLchAShxQ_001709": {
    "video_start": 1709,
    "video_end": 1782,
    "anomaly_start": 34,
    "anomaly_end": 69,
    "anomaly_class": "ego: turning",
    "num_frames": 74,
    "subset": "val"
  },
  "cfrLchAShxQ_002469": {
    "video_start": 2469,
    "video_end": 2575,
    "anomaly_start": 16,
    "anomaly_end": 79,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 107,
    "subset": "val"
  },
  "cfrLchAShxQ_002859": {
    "video_start": 2859,
    "video_end": 2942,
    "anomaly_start": 48,
    "anomaly_end": 70,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 84,
    "subset": "val"
  },
  "cfrLchAShxQ_003406": {
    "video_start": 3406,
    "video_end": 3555,
    "anomaly_start": 1,
    "anomaly_end": 104,
    "anomaly_class": "ego: lateral",
    "num_frames": 150,
    "subset": "val"
  },
  "cfrLchAShxQ_003557": {
    "video_start": 3557,
    "video_end": 3643,
    "anomaly_start": 9,
    "anomaly_end": 69,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 87,
    "subset": "val"
  },
  "cfrLchAShxQ_003645": {
    "video_start": 3645,
    "video_end": 3716,
    "anomaly_start": 26,
    "anomaly_end": 52,
    "anomaly_class": "other: turning",
    "num_frames": 72,
    "subset": "val"
  },
  "cfrLchAShxQ_003718": {
    "video_start": 3718,
    "video_end": 3803,
    "anomaly_start": 41,
    "anomaly_end": 67,
    "anomaly_class": "other: obstacle",
    "num_frames": 86,
    "subset": "val"
  },
  "d2SCftR5sWc_000191": {
    "video_start": 191,
    "video_end": 299,
    "anomaly_start": 36,
    "anomaly_end": 76,
    "anomaly_class": "ego: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "d2SCftR5sWc_001735": {
    "video_start": 1735,
    "video_end": 1843,
    "anomaly_start": 31,
    "anomaly_end": 56,
    "anomaly_class": "ego: oncoming",
    "num_frames": 109,
    "subset": "val"
  },
  "d2SCftR5sWc_002095": {
    "video_start": 2095,
    "video_end": 2194,
    "anomaly_start": 63,
    "anomaly_end": 80,
    "anomaly_class": "ego: pedestrian",
    "num_frames": 100,
    "subset": "val"
  },
  "d2SCftR5sWc_002296": {
    "video_start": 2296,
    "video_end": 2394,
    "anomaly_start": 27,
    "anomaly_end": 57,
    "anomaly_class": "ego: pedestrian",
    "num_frames": 99,
    "subset": "val"
  },
  "d2SCftR5sWc_002396": {
    "video_start": 2396,
    "video_end": 2514,
    "anomaly_start": 41,
    "anomaly_end": 119,
    "anomaly_class": "other: turning",
    "num_frames": 119,
    "subset": "val"
  },
  "d2SCftR5sWc_003882": {
    "video_start": 3882,
    "video_end": 3980,
    "anomaly_start": 21,
    "anomaly_end": 97,
    "anomaly_class": "ego: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "d2SCftR5sWc_003982": {
    "video_start": 3982,
    "video_end": 4070,
    "anomaly_start": 21,
    "anomaly_end": 51,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 89,
    "subset": "val"
  },
  "d2SCftR5sWc_004072": {
    "video_start": 4072,
    "video_end": 4171,
    "anomaly_start": 34,
    "anomaly_end": 66,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 100,
    "subset": "val"
  },
  "d2SCftR5sWc_004399": {
    "video_start": 4399,
    "video_end": 4487,
    "anomaly_start": 17,
    "anomaly_end": 38,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 89,
    "subset": "val"
  },
  "d2SCftR5sWc_004489": {
    "video_start": 4489,
    "video_end": 4587,
    "anomaly_start": 47,
    "anomaly_end": 72,
    "anomaly_class": "other: lateral",
    "num_frames": 99,
    "subset": "val"
  },
  "dZG1VKsgs60_000582": {
    "video_start": 582,
    "video_end": 700,
    "anomaly_start": 32,
    "anomaly_end": 79,
    "anomaly_class": "other: lateral",
    "num_frames": 119,
    "subset": "val"
  },
  "dZG1VKsgs60_000825": {
    "video_start": 825,
    "video_end": 934,
    "anomaly_start": 53,
    "anomaly_end": 102,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 110,
    "subset": "val"
  },
  "dZG1VKsgs60_001293": {
    "video_start": 1293,
    "video_end": 1451,
    "anomaly_start": 53,
    "anomaly_end": 75,
    "anomaly_class": "other: turning",
    "num_frames": 159,
    "subset": "val"
  },
  "dZG1VKsgs60_001453": {
    "video_start": 1453,
    "video_end": 1561,
    "anomaly_start": 55,
    "anomaly_end": 109,
    "anomaly_class": "ego: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "dZG1VKsgs60_002129": {
    "video_start": 2129,
    "video_end": 2264,
    "anomaly_start": 74,
    "anomaly_end": 95,
    "anomaly_class": "ego: oncoming",
    "num_frames": 136,
    "subset": "val"
  },
  "dZG1VKsgs60_002494": {
    "video_start": 2494,
    "video_end": 2586,
    "anomaly_start": 46,
    "anomaly_end": 60,
    "anomaly_class": "ego: turning",
    "num_frames": 93,
    "subset": "val"
  },
  "dZG1VKsgs60_002768": {
    "video_start": 2768,
    "video_end": 2867,
    "anomaly_start": 32,
    "anomaly_end": 78,
    "anomaly_class": "ego: turning",
    "num_frames": 100,
    "subset": "val"
  },
  "dZG1VKsgs60_004008": {
    "video_start": 4008,
    "video_end": 4183,
    "anomaly_start": 21,
    "anomaly_end": 65,
    "anomaly_class": "other: pedestrian",
    "num_frames": 176,
    "subset": "val"
  },
  "dZG1VKsgs60_004721": {
    "video_start": 4721,
    "video_end": 4809,
    "anomaly_start": 22,
    "anomaly_end": 68,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 89,
    "subset": "val"
  },
  "dZG1VKsgs60_004811": {
    "video_start": 4811,
    "video_end": 4870,
    "anomaly_start": 6,
    "anomaly_end": 48,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 60,
    "subset": "val"
  },
  "dZG1VKsgs60_005432": {
    "video_start": 5432,
    "video_end": 5555,
    "anomaly_start": 82,
    "anomaly_end": 107,
    "anomaly_class": "other: turning",
    "num_frames": 124,
    "subset": "val"
  },
  "e3hXj7S3QyA_000301": {
    "video_start": 301,
    "video_end": 353,
    "anomaly_start": 31,
    "anomaly_end": 53,
    "anomaly_class": "ego: oncoming",
    "num_frames": 53,
    "subset": "val"
  },
  "e3hXj7S3QyA_000706": {
    "video_start": 706,
    "video_end": 760,
    "anomaly_start": 16,
    "anomaly_end": 53,
    "anomaly_class": "other: turning",
    "num_frames": 55,
    "subset": "val"
  },
  "e3hXj7S3QyA_001748": {
    "video_start": 1748,
    "video_end": 1804,
    "anomaly_start": 27,
    "anomaly_end": 57,
    "anomaly_class": "ego: turning",
    "num_frames": 57,
    "subset": "val"
  },
  "eA-kbOZ1_sc_000668": {
    "video_start": 668,
    "video_end": 768,
    "anomaly_start": 42,
    "anomaly_end": 78,
    "anomaly_class": "other: lateral",
    "num_frames": 101,
    "subset": "val"
  },
  "eA-kbOZ1_sc_003253": {
    "video_start": 3253,
    "video_end": 3313,
    "anomaly_start": 12,
    "anomaly_end": 54,
    "anomaly_class": "ego: oncoming",
    "num_frames": 61,
    "subset": "val"
  },
  "eA-kbOZ1_sc_005273": {
    "video_start": 5273,
    "video_end": 5333,
    "anomaly_start": 24,
    "anomaly_end": 50,
    "anomaly_class": "other: turning",
    "num_frames": 61,
    "subset": "val"
  },
  "eN6y6-4C1Rc_000267": {
    "video_start": 267,
    "video_end": 381,
    "anomaly_start": 36,
    "anomaly_end": 43,
    "anomaly_class": "ego: lateral",
    "num_frames": 115,
    "subset": "val"
  },
  "eN6y6-4C1Rc_000387": {
    "video_start": 387,
    "video_end": 476,
    "anomaly_start": 27,
    "anomaly_end": 72,
    "anomaly_class": "ego: turning",
    "num_frames": 90,
    "subset": "val"
  },
  "eN6y6-4C1Rc_001527": {
    "video_start": 1527,
    "video_end": 1636,
    "anomaly_start": 18,
    "anomaly_end": 82,
    "anomaly_class": "ego: lateral",
    "num_frames": 110,
    "subset": "val"
  },
  "eN6y6-4C1Rc_001843": {
    "video_start": 1843,
    "video_end": 1936,
    "anomaly_start": 43,
    "anomaly_end": 72,
    "anomaly_class": "ego: turning",
    "num_frames": 94,
    "subset": "val"
  },
  "eN6y6-4C1Rc_003465": {
    "video_start": 3465,
    "video_end": 3565,
    "anomaly_start": 28,
    "anomaly_end": 45,
    "anomaly_class": "other: lateral",
    "num_frames": 101,
    "subset": "val"
  },
  "eN6y6-4C1Rc_003571": {
    "video_start": 3571,
    "video_end": 3676,
    "anomaly_start": 42,
    "anomaly_end": 98,
    "anomaly_class": "ego: turning",
    "num_frames": 106,
    "subset": "val"
  },
  "eN6y6-4C1Rc_003883": {
    "video_start": 3883,
    "video_end": 3957,
    "anomaly_start": 31,
    "anomaly_end": 49,
    "anomaly_class": "other: turning",
    "num_frames": 75,
    "subset": "val"
  },
  "eN6y6-4C1Rc_004495": {
    "video_start": 4495,
    "video_end": 4573,
    "anomaly_start": 11,
    "anomaly_end": 32,
    "anomaly_class": "ego: lateral",
    "num_frames": 79,
    "subset": "val"
  },
  "eN6y6-4C1Rc_005283": {
    "video_start": 5283,
    "video_end": 5394,
    "anomaly_start": 59,
    "anomaly_end": 85,
    "anomaly_class": "ego: turning",
    "num_frames": 112,
    "subset": "val"
  },
  "eN6y6-4C1Rc_005830": {
    "video_start": 5830,
    "video_end": 5888,
    "anomaly_start": 12,
    "anomaly_end": 31,
    "anomaly_class": "other: obstacle",
    "num_frames": 59,
    "subset": "val"
  },
  "eN6y6-4C1Rc_005894": {
    "video_start": 5894,
    "video_end": 5962,
    "anomaly_start": 19,
    "anomaly_end": 47,
    "anomaly_class": "ego: turning",
    "num_frames": 69,
    "subset": "val"
  },
  "eN6y6-4C1Rc_005967": {
    "video_start": 5967,
    "video_end": 6035,
    "anomaly_start": 23,
    "anomaly_end": 36,
    "anomaly_class": "ego: turning",
    "num_frames": 69,
    "subset": "val"
  },
  "eW9vWga9F5M_000711": {
    "video_start": 711,
    "video_end": 799,
    "anomaly_start": 28,
    "anomaly_end": 58,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "eWWgJznGg6U_000422": {
    "video_start": 422,
    "video_end": 524,
    "anomaly_start": 37,
    "anomaly_end": 103,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 103,
    "subset": "val"
  },
  "eWWgJznGg6U_000731": {
    "video_start": 731,
    "video_end": 788,
    "anomaly_start": 13,
    "anomaly_end": 52,
    "anomaly_class": "ego: turning",
    "num_frames": 58,
    "subset": "val"
  },
  "eWWgJznGg6U_003975": {
    "video_start": 3975,
    "video_end": 4065,
    "anomaly_start": 34,
    "anomaly_end": 60,
    "anomaly_class": "ego: turning",
    "num_frames": 91,
    "subset": "val"
  },
  "eWWgJznGg6U_004947": {
    "video_start": 4947,
    "video_end": 5024,
    "anomaly_start": 35,
    "anomaly_end": 72,
    "anomaly_class": "other: turning",
    "num_frames": 78,
    "subset": "val"
  },
  "eWWgJznGg6U_006506": {
    "video_start": 6506,
    "video_end": 6666,
    "anomaly_start": 67,
    "anomaly_end": 116,
    "anomaly_class": "other: lateral",
    "num_frames": 161,
    "subset": "val"
  },
  "eWWgJznGg6U_006687": {
    "video_start": 6687,
    "video_end": 6738,
    "anomaly_start": 19,
    "anomaly_end": 40,
    "anomaly_class": "other: turning",
    "num_frames": 52,
    "subset": "val"
  },
  "ecRog9N1Qf0_001036": {
    "video_start": 1036,
    "video_end": 1150,
    "anomaly_start": 20,
    "anomaly_end": 46,
    "anomaly_class": "ego: unknown",
    "num_frames": 115,
    "subset": "val"
  },
  "ecRog9N1Qf0_002116": {
    "video_start": 2116,
    "video_end": 2203,
    "anomaly_start": 32,
    "anomaly_end": 51,
    "anomaly_class": "ego: pedestrian",
    "num_frames": 88,
    "subset": "val"
  },
  "ecRog9N1Qf0_004091": {
    "video_start": 4091,
    "video_end": 4157,
    "anomaly_start": 11,
    "anomaly_end": 43,
    "anomaly_class": "ego: lateral",
    "num_frames": 67,
    "subset": "val"
  },
  "ezY7QrSWeWw_000112": {
    "video_start": 112,
    "video_end": 188,
    "anomaly_start": 22,
    "anomaly_end": 61,
    "anomaly_class": "ego: oncoming",
    "num_frames": 77,
    "subset": "val"
  },
  "ezY7QrSWeWw_000837": {
    "video_start": 837,
    "video_end": 907,
    "anomaly_start": 36,
    "anomaly_end": 62,
    "anomaly_class": "other: turning",
    "num_frames": 71,
    "subset": "val"
  },
  "ezY7QrSWeWw_001369": {
    "video_start": 1369,
    "video_end": 1418,
    "anomaly_start": 10,
    "anomaly_end": 35,
    "anomaly_class": "ego: turning",
    "num_frames": 50,
    "subset": "val"
  },
  "ezY7QrSWeWw_001606": {
    "video_start": 1606,
    "video_end": 1664,
    "anomaly_start": 18,
    "anomaly_end": 55,
    "anomaly_class": "ego: turning",
    "num_frames": 59,
    "subset": "val"
  },
  "ezY7QrSWeWw_001685": {
    "video_start": 1685,
    "video_end": 1752,
    "anomaly_start": 23,
    "anomaly_end": 24,
    "anomaly_class": "other: unknown",
    "num_frames": 68,
    "subset": "val"
  },
  "f482qwPz7ns_000644": {
    "video_start": 644,
    "video_end": 772,
    "anomaly_start": 57,
    "anomaly_end": 89,
    "anomaly_class": "ego: turning",
    "num_frames": 129,
    "subset": "val"
  },
  "f482qwPz7ns_000774": {
    "video_start": 774,
    "video_end": 914,
    "anomaly_start": 74,
    "anomaly_end": 120,
    "anomaly_class": "ego: turning",
    "num_frames": 141,
    "subset": "val"
  },
  "f482qwPz7ns_000916": {
    "video_start": 916,
    "video_end": 1004,
    "anomaly_start": 41,
    "anomaly_end": 66,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "f482qwPz7ns_002749": {
    "video_start": 2749,
    "video_end": 2845,
    "anomaly_start": 43,
    "anomaly_end": 70,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 97,
    "subset": "val"
  },
  "f482qwPz7ns_002847": {
    "video_start": 2847,
    "video_end": 2967,
    "anomaly_start": 37,
    "anomaly_end": 95,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 121,
    "subset": "val"
  },
  "f482qwPz7ns_003534": {
    "video_start": 3534,
    "video_end": 3648,
    "anomaly_start": 56,
    "anomaly_end": 115,
    "anomaly_class": "ego: turning",
    "num_frames": 115,
    "subset": "val"
  },
  "f482qwPz7ns_003922": {
    "video_start": 3922,
    "video_end": 4010,
    "anomaly_start": 23,
    "anomaly_end": 58,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "f482qwPz7ns_005487": {
    "video_start": 5487,
    "video_end": 5593,
    "anomaly_start": 46,
    "anomaly_end": 81,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 107,
    "subset": "val"
  },
  "fHUJcUaW2mE_000920": {
    "video_start": 920,
    "video_end": 1018,
    "anomaly_start": 34,
    "anomaly_end": 90,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 99,
    "subset": "val"
  },
  "fHUJcUaW2mE_001020": {
    "video_start": 1020,
    "video_end": 1101,
    "anomaly_start": 39,
    "anomaly_end": 56,
    "anomaly_class": "ego: turning",
    "num_frames": 82,
    "subset": "val"
  },
  "fHUJcUaW2mE_001313": {
    "video_start": 1313,
    "video_end": 1431,
    "anomaly_start": 64,
    "anomaly_end": 87,
    "anomaly_class": "ego: oncoming",
    "num_frames": 119,
    "subset": "val"
  },
  "fHUJcUaW2mE_001714": {
    "video_start": 1714,
    "video_end": 1923,
    "anomaly_start": 34,
    "anomaly_end": 192,
    "anomaly_class": "ego: lateral",
    "num_frames": 210,
    "subset": "val"
  },
  "fHUJcUaW2mE_002364": {
    "video_start": 2364,
    "video_end": 2503,
    "anomaly_start": 37,
    "anomaly_end": 125,
    "anomaly_class": "ego: lateral",
    "num_frames": 140,
    "subset": "val"
  },
  "fHUJcUaW2mE_002702": {
    "video_start": 2702,
    "video_end": 2818,
    "anomaly_start": 13,
    "anomaly_end": 59,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 117,
    "subset": "val"
  },
  "fHUJcUaW2mE_002820": {
    "video_start": 2820,
    "video_end": 2921,
    "anomaly_start": 47,
    "anomaly_end": 73,
    "anomaly_class": "ego: oncoming",
    "num_frames": 102,
    "subset": "val"
  },
  "fHUJcUaW2mE_003090": {
    "video_start": 3090,
    "video_end": 3189,
    "anomaly_start": 29,
    "anomaly_end": 64,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 100,
    "subset": "val"
  },
  "fHUJcUaW2mE_003191": {
    "video_start": 3191,
    "video_end": 3259,
    "anomaly_start": 35,
    "anomaly_end": 53,
    "anomaly_class": "other: turning",
    "num_frames": 69,
    "subset": "val"
  },
  "fHUJcUaW2mE_003261": {
    "video_start": 3261,
    "video_end": 3380,
    "anomaly_start": 78,
    "anomaly_end": 120,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 120,
    "subset": "val"
  },
  "fTmYppzzloc_000821": {
    "video_start": 821,
    "video_end": 870,
    "anomaly_start": 2,
    "anomaly_end": 49,
    "anomaly_class": "other: lateral",
    "num_frames": 50,
    "subset": "val"
  },
  "fWJbp43k644_000051": {
    "video_start": 51,
    "video_end": 129,
    "anomaly_start": 25,
    "anomaly_end": 42,
    "anomaly_class": "other: unknown",
    "num_frames": 79,
    "subset": "val"
  },
  "fWJbp43k644_000436": {
    "video_start": 436,
    "video_end": 531,
    "anomaly_start": 39,
    "anomaly_end": 57,
    "anomaly_class": "ego: turning",
    "num_frames": 96,
    "subset": "val"
  },
  "fWJbp43k644_000657": {
    "video_start": 657,
    "video_end": 747,
    "anomaly_start": 32,
    "anomaly_end": 54,
    "anomaly_class": "ego: oncoming",
    "num_frames": 91,
    "subset": "val"
  },
  "fWJbp43k644_001197": {
    "video_start": 1197,
    "video_end": 1323,
    "anomaly_start": 33,
    "anomaly_end": 72,
    "anomaly_class": "ego: turning",
    "num_frames": 127,
    "subset": "val"
  },
  "fWJbp43k644_001505": {
    "video_start": 1505,
    "video_end": 1585,
    "anomaly_start": 37,
    "anomaly_end": 69,
    "anomaly_class": "ego: turning",
    "num_frames": 81,
    "subset": "val"
  },
  "fWJbp43k644_001672": {
    "video_start": 1672,
    "video_end": 1761,
    "anomaly_start": 34,
    "anomaly_end": 58,
    "anomaly_class": "ego: turning",
    "num_frames": 90,
    "subset": "val"
  },
  "fWJbp43k644_002658": {
    "video_start": 2658,
    "video_end": 2779,
    "anomaly_start": 42,
    "anomaly_end": 121,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 122,
    "subset": "val"
  },
  "fWJbp43k644_004034": {
    "video_start": 4034,
    "video_end": 4134,
    "anomaly_start": 37,
    "anomaly_end": 72,
    "anomaly_class": "ego: oncoming",
    "num_frames": 101,
    "subset": "val"
  },
  "fWJbp43k644_004136": {
    "video_start": 4136,
    "video_end": 4273,
    "anomaly_start": 26,
    "anomaly_end": 51,
    "anomaly_class": "ego: lateral",
    "num_frames": 138,
    "subset": "val"
  },
  "fWJbp43k644_005053": {
    "video_start": 5053,
    "video_end": 5153,
    "anomaly_start": 41,
    "anomaly_end": 61,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 101,
    "subset": "val"
  },
  "fWJbp43k644_005522": {
    "video_start": 5522,
    "video_end": 5679,
    "anomaly_start": 13,
    "anomaly_end": 52,
    "anomaly_class": "ego: lateral",
    "num_frames": 158,
    "subset": "val"
  },
  "fb5i7JlmJdg_000923": {
    "video_start": 923,
    "video_end": 1071,
    "anomaly_start": 73,
    "anomaly_end": 110,
    "anomaly_class": "ego: turning",
    "num_frames": 149,
    "subset": "val"
  },
  "fb5i7JlmJdg_001434": {
    "video_start": 1434,
    "video_end": 1522,
    "anomaly_start": 39,
    "anomaly_end": 55,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "fb5i7JlmJdg_003069": {
    "video_start": 3069,
    "video_end": 3217,
    "anomaly_start": 40,
    "anomaly_end": 133,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 149,
    "subset": "val"
  },
  "fb5i7JlmJdg_004249": {
    "video_start": 4249,
    "video_end": 4347,
    "anomaly_start": 50,
    "anomaly_end": 68,
    "anomaly_class": "ego: start_stop_or_stationary",
    "num_frames": 99,
    "subset": "val"
  },
  "fb5i7JlmJdg_004349": {
    "video_start": 4349,
    "video_end": 4447,
    "anomaly_start": 45,
    "anomaly_end": 67,
    "anomaly_class": "ego: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "fb5i7JlmJdg_004952": {
    "video_start": 4952,
    "video_end": 5050,
    "anomaly_start": 44,
    "anomaly_end": 57,
    "anomaly_class": "other: oncoming",
    "num_frames": 99,
    "subset": "val"
  },
  "fb5i7JlmJdg_005362": {
    "video_start": 5362,
    "video_end": 5440,
    "anomaly_start": 34,
    "anomaly_end": 56,
    "anomaly_class": "other: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "fb5i7JlmJdg_005442": {
    "video_start": 5442,
    "video_end": 5551,
    "anomaly_start": 58,
    "anomaly_end": 80,
    "anomaly_class": "ego: oncoming",
    "num_frames": 110,
    "subset": "val"
  },
  "fdvMUP8qvzw_001014": {
    "video_start": 1014,
    "video_end": 1104,
    "anomaly_start": 28,
    "anomaly_end": 51,
    "anomaly_class": "ego: oncoming",
    "num_frames": 91,
    "subset": "val"
  },
  "fdvMUP8qvzw_001397": {
    "video_start": 1397,
    "video_end": 1505,
    "anomaly_start": 45,
    "anomaly_end": 62,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 109,
    "subset": "val"
  },
  "fdvMUP8qvzw_004232": {
    "video_start": 4232,
    "video_end": 4291,
    "anomaly_start": 25,
    "anomaly_end": 41,
    "anomaly_class": "other: turning",
    "num_frames": 60,
    "subset": "val"
  },
  "fdvMUP8qvzw_004293": {
    "video_start": 4293,
    "video_end": 4370,
    "anomaly_start": 31,
    "anomaly_end": 53,
    "anomaly_class": "ego: turning",
    "num_frames": 78,
    "subset": "val"
  },
  "fdvMUP8qvzw_005495": {
    "video_start": 5495,
    "video_end": 5593,
    "anomaly_start": 20,
    "anomaly_end": 69,
    "anomaly_class": "ego: unknown",
    "num_frames": 99,
    "subset": "val"
  },
  "fsqORUrKLYg_001225": {
    "video_start": 1225,
    "video_end": 1333,
    "anomaly_start": 64,
    "anomaly_end": 86,
    "anomaly_class": "other: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "fsqORUrKLYg_003730": {
    "video_start": 3730,
    "video_end": 3848,
    "anomaly_start": 40,
    "anomaly_end": 59,
    "anomaly_class": "ego: oncoming",
    "num_frames": 119,
    "subset": "val"
  },
  "fsqORUrKLYg_004820": {
    "video_start": 4820,
    "video_end": 4918,
    "anomaly_start": 37,
    "anomaly_end": 71,
    "anomaly_class": "ego: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "gDjeTHGWX9c_000403": {
    "video_start": 403,
    "video_end": 510,
    "anomaly_start": 37,
    "anomaly_end": 84,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 108,
    "subset": "val"
  },
  "gDjeTHGWX9c_000656": {
    "video_start": 656,
    "video_end": 735,
    "anomaly_start": 7,
    "anomaly_end": 37,
    "anomaly_class": "ego: lateral",
    "num_frames": 80,
    "subset": "val"
  },
  "gDjeTHGWX9c_001476": {
    "video_start": 1476,
    "video_end": 1545,
    "anomaly_start": 32,
    "anomaly_end": 70,
    "anomaly_class": "ego: obstacle",
    "num_frames": 70,
    "subset": "val"
  },
  "gDjeTHGWX9c_001707": {
    "video_start": 1707,
    "video_end": 1794,
    "anomaly_start": 36,
    "anomaly_end": 64,
    "anomaly_class": "ego: lateral",
    "num_frames": 88,
    "subset": "val"
  },
  "gDjeTHGWX9c_002793": {
    "video_start": 2793,
    "video_end": 2872,
    "anomaly_start": 9,
    "anomaly_end": 27,
    "anomaly_class": "other: turning",
    "num_frames": 80,
    "subset": "val"
  },
  "gDjeTHGWX9c_003170": {
    "video_start": 3170,
    "video_end": 3244,
    "anomaly_start": 20,
    "anomaly_end": 58,
    "anomaly_class": "ego: turning",
    "num_frames": 75,
    "subset": "val"
  },
  "gDjeTHGWX9c_003429": {
    "video_start": 3429,
    "video_end": 3508,
    "anomaly_start": 25,
    "anomaly_end": 66,
    "anomaly_class": "ego: lateral",
    "num_frames": 80,
    "subset": "val"
  },
  "gDjeTHGWX9c_003510": {
    "video_start": 3510,
    "video_end": 3599,
    "anomaly_start": 34,
    "anomaly_end": 58,
    "anomaly_class": "ego: oncoming",
    "num_frames": 90,
    "subset": "val"
  },
  "gDjeTHGWX9c_003722": {
    "video_start": 3722,
    "video_end": 3818,
    "anomaly_start": 44,
    "anomaly_end": 91,
    "anomaly_class": "ego: lateral",
    "num_frames": 97,
    "subset": "val"
  },
  "gDjeTHGWX9c_004036": {
    "video_start": 4036,
    "video_end": 4104,
    "anomaly_start": 25,
    "anomaly_end": 42,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 69,
    "subset": "val"
  },
  "gDjeTHGWX9c_004634": {
    "video_start": 4634,
    "video_end": 4690,
    "anomaly_start": 22,
    "anomaly_end": 56,
    "anomaly_class": "ego: lateral",
    "num_frames": 57,
    "subset": "val"
  },
  "gDjeTHGWX9c_005264": {
    "video_start": 5264,
    "video_end": 5340,
    "anomaly_start": 24,
    "anomaly_end": 50,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 77,
    "subset": "val"
  },
  "gDjeTHGWX9c_005826": {
    "video_start": 5826,
    "video_end": 5912,
    "anomaly_start": 38,
    "anomaly_end": 83,
    "anomaly_class": "ego: lateral",
    "num_frames": 87,
    "subset": "val"
  },
  "gDjeTHGWX9c_006247": {
    "video_start": 6247,
    "video_end": 6346,
    "anomaly_start": 38,
    "anomaly_end": 83,
    "anomaly_class": "other: obstacle",
    "num_frames": 100,
    "subset": "val"
  },
  "gXezhrOijmQ_000311": {
    "video_start": 311,
    "video_end": 399,
    "anomaly_start": 48,
    "anomaly_end": 73,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 89,
    "subset": "val"
  },
  "gXezhrOijmQ_000949": {
    "video_start": 949,
    "video_end": 1057,
    "anomaly_start": 38,
    "anomaly_end": 76,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 109,
    "subset": "val"
  },
  "gXezhrOijmQ_001059": {
    "video_start": 1059,
    "video_end": 1152,
    "anomaly_start": 37,
    "anomaly_end": 94,
    "anomaly_class": "other: lateral",
    "num_frames": 94,
    "subset": "val"
  },
  "gXezhrOijmQ_001448": {
    "video_start": 1448,
    "video_end": 1526,
    "anomaly_start": 31,
    "anomaly_end": 79,
    "anomaly_class": "ego: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "gXezhrOijmQ_003917": {
    "video_start": 3917,
    "video_end": 4035,
    "anomaly_start": 30,
    "anomaly_end": 64,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 119,
    "subset": "val"
  },
  "gXezhrOijmQ_004037": {
    "video_start": 4037,
    "video_end": 4125,
    "anomaly_start": 25,
    "anomaly_end": 89,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 89,
    "subset": "val"
  },
  "gZbGLa253Ak_000073": {
    "video_start": 73,
    "video_end": 219,
    "anomaly_start": 49,
    "anomaly_end": 110,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 147,
    "subset": "val"
  },
  "gZbGLa253Ak_000341": {
    "video_start": 341,
    "video_end": 429,
    "anomaly_start": 39,
    "anomaly_end": 77,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "gZbGLa253Ak_000932": {
    "video_start": 932,
    "video_end": 1040,
    "anomaly_start": 49,
    "anomaly_end": 83,
    "anomaly_class": "other: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "gZbGLa253Ak_001252": {
    "video_start": 1252,
    "video_end": 1360,
    "anomaly_start": 42,
    "anomaly_end": 68,
    "anomaly_class": "other: obstacle",
    "num_frames": 109,
    "subset": "val"
  },
  "gZbGLa253Ak_001583": {
    "video_start": 1583,
    "video_end": 1799,
    "anomaly_start": 79,
    "anomaly_end": 141,
    "anomaly_class": "other: turning",
    "num_frames": 217,
    "subset": "val"
  },
  "gZbGLa253Ak_001801": {
    "video_start": 1801,
    "video_end": 1879,
    "anomaly_start": 33,
    "anomaly_end": 79,
    "anomaly_class": "ego: unknown",
    "num_frames": 79,
    "subset": "val"
  },
  "gZbGLa253Ak_001961": {
    "video_start": 1961,
    "video_end": 2049,
    "anomaly_start": 46,
    "anomaly_end": 87,
    "anomaly_class": "ego: unknown",
    "num_frames": 89,
    "subset": "val"
  },
  "gZbGLa253Ak_002051": {
    "video_start": 2051,
    "video_end": 2189,
    "anomaly_start": 82,
    "anomaly_end": 122,
    "anomaly_class": "other: unknown",
    "num_frames": 139,
    "subset": "val"
  },
  "gZbGLa253Ak_002697": {
    "video_start": 2697,
    "video_end": 2815,
    "anomaly_start": 65,
    "anomaly_end": 104,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 119,
    "subset": "val"
  },
  "gZbGLa253Ak_002817": {
    "video_start": 2817,
    "video_end": 2905,
    "anomaly_start": 40,
    "anomaly_end": 72,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "gZbGLa253Ak_003015": {
    "video_start": 3015,
    "video_end": 3078,
    "anomaly_start": 48,
    "anomaly_end": 64,
    "anomaly_class": "ego: oncoming",
    "num_frames": 64,
    "subset": "val"
  },
  "gZbGLa253Ak_003103": {
    "video_start": 3103,
    "video_end": 3204,
    "anomaly_start": 56,
    "anomaly_end": 94,
    "anomaly_class": "other: turning",
    "num_frames": 102,
    "subset": "val"
  },
  "gZbGLa253Ak_003759": {
    "video_start": 3759,
    "video_end": 3847,
    "anomaly_start": 13,
    "anomaly_end": 66,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "gZbGLa253Ak_003975": {
    "video_start": 3975,
    "video_end": 4073,
    "anomaly_start": 44,
    "anomaly_end": 68,
    "anomaly_class": "ego: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "gZbGLa253Ak_004075": {
    "video_start": 4075,
    "video_end": 4165,
    "anomaly_start": 42,
    "anomaly_end": 66,
    "anomaly_class": "ego: oncoming",
    "num_frames": 91,
    "subset": "val"
  },
  "gZbGLa253Ak_004754": {
    "video_start": 4754,
    "video_end": 4852,
    "anomaly_start": 49,
    "anomaly_end": 73,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 99,
    "subset": "val"
  },
  "gZbGLa253Ak_004854": {
    "video_start": 4854,
    "video_end": 4972,
    "anomaly_start": 58,
    "anomaly_end": 119,
    "anomaly_class": "ego: lateral",
    "num_frames": 119,
    "subset": "val"
  },
  "h4OXsRSWLLc_000155": {
    "video_start": 155,
    "video_end": 229,
    "anomaly_start": 8,
    "anomaly_end": 42,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 75,
    "subset": "val"
  },
  "h4OXsRSWLLc_000550": {
    "video_start": 550,
    "video_end": 612,
    "anomaly_start": 23,
    "anomaly_end": 58,
    "anomaly_class": "other: obstacle",
    "num_frames": 63,
    "subset": "val"
  },
  "h55PiQMnlJY_000824": {
    "video_start": 824,
    "video_end": 932,
    "anomaly_start": 41,
    "anomaly_end": 66,
    "anomaly_class": "other: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "h55PiQMnlJY_001034": {
    "video_start": 1034,
    "video_end": 1130,
    "anomaly_start": 30,
    "anomaly_end": 66,
    "anomaly_class": "other: oncoming",
    "num_frames": 97,
    "subset": "val"
  },
  "h55PiQMnlJY_001860": {
    "video_start": 1860,
    "video_end": 1978,
    "anomaly_start": 61,
    "anomaly_end": 92,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 119,
    "subset": "val"
  },
  "h55PiQMnlJY_002326": {
    "video_start": 2326,
    "video_end": 2424,
    "anomaly_start": 41,
    "anomaly_end": 69,
    "anomaly_class": "ego: oncoming",
    "num_frames": 99,
    "subset": "val"
  },
  "h55PiQMnlJY_002426": {
    "video_start": 2426,
    "video_end": 2569,
    "anomaly_start": 69,
    "anomaly_end": 110,
    "anomaly_class": "other: turning",
    "num_frames": 144,
    "subset": "val"
  },
  "h55PiQMnlJY_003085": {
    "video_start": 3085,
    "video_end": 3173,
    "anomaly_start": 48,
    "anomaly_end": 69,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "h55PiQMnlJY_003472": {
    "video_start": 3472,
    "video_end": 3550,
    "anomaly_start": 36,
    "anomaly_end": 63,
    "anomaly_class": "other: pedestrian",
    "num_frames": 79,
    "subset": "val"
  },
  "h55PiQMnlJY_003552": {
    "video_start": 3552,
    "video_end": 3673,
    "anomaly_start": 37,
    "anomaly_end": 72,
    "anomaly_class": "other: pedestrian",
    "num_frames": 122,
    "subset": "val"
  },
  "h55PiQMnlJY_004143": {
    "video_start": 4143,
    "video_end": 4301,
    "anomaly_start": 56,
    "anomaly_end": 144,
    "anomaly_class": "other: turning",
    "num_frames": 159,
    "subset": "val"
  },
  "ha-IeID24As_000340": {
    "video_start": 340,
    "video_end": 445,
    "anomaly_start": 37,
    "anomaly_end": 83,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 106,
    "subset": "val"
  },
  "ha-IeID24As_001147": {
    "video_start": 1147,
    "video_end": 1245,
    "anomaly_start": 58,
    "anomaly_end": 74,
    "anomaly_class": "other: obstacle",
    "num_frames": 99,
    "subset": "val"
  },
  "ha-IeID24As_001466": {
    "video_start": 1466,
    "video_end": 1565,
    "anomaly_start": 32,
    "anomaly_end": 53,
    "anomaly_class": "other: turning",
    "num_frames": 100,
    "subset": "val"
  },
  "ha-IeID24As_003932": {
    "video_start": 3932,
    "video_end": 4010,
    "anomaly_start": 34,
    "anomaly_end": 68,
    "anomaly_class": "ego: lateral",
    "num_frames": 79,
    "subset": "val"
  },
  "ha-IeID24As_004213": {
    "video_start": 4213,
    "video_end": 4301,
    "anomaly_start": 35,
    "anomaly_end": 53,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "ha-IeID24As_004799": {
    "video_start": 4799,
    "video_end": 4877,
    "anomaly_start": 26,
    "anomaly_end": 45,
    "anomaly_class": "other: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "ha-IeID24As_005456": {
    "video_start": 5456,
    "video_end": 5534,
    "anomaly_start": 31,
    "anomaly_end": 54,
    "anomaly_class": "ego: pedestrian",
    "num_frames": 79,
    "subset": "val"
  },
  "ha-IeID24As_005536": {
    "video_start": 5536,
    "video_end": 5654,
    "anomaly_start": 30,
    "anomaly_end": 49,
    "anomaly_class": "other: turning",
    "num_frames": 119,
    "subset": "val"
  },
  "hdOtS-BpUPY_000303": {
    "video_start": 303,
    "video_end": 428,
    "anomaly_start": 65,
    "anomaly_end": 103,
    "anomaly_class": "ego: oncoming",
    "num_frames": 126,
    "subset": "val"
  },
  "hdOtS-BpUPY_000430": {
    "video_start": 430,
    "video_end": 548,
    "anomaly_start": 43,
    "anomaly_end": 85,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 119,
    "subset": "val"
  },
  "hdOtS-BpUPY_000700": {
    "video_start": 700,
    "video_end": 818,
    "anomaly_start": 36,
    "anomaly_end": 113,
    "anomaly_class": "other: turning",
    "num_frames": 119,
    "subset": "val"
  },
  "hdOtS-BpUPY_001450": {
    "video_start": 1450,
    "video_end": 1549,
    "anomaly_start": 50,
    "anomaly_end": 71,
    "anomaly_class": "other: oncoming",
    "num_frames": 100,
    "subset": "val"
  },
  "hdOtS-BpUPY_002106": {
    "video_start": 2106,
    "video_end": 2203,
    "anomaly_start": 30,
    "anomaly_end": 62,
    "anomaly_class": "other: turning",
    "num_frames": 98,
    "subset": "val"
  },
  "hdOtS-BpUPY_002777": {
    "video_start": 2777,
    "video_end": 2859,
    "anomaly_start": 42,
    "anomaly_end": 65,
    "anomaly_class": "ego: unknown",
    "num_frames": 83,
    "subset": "val"
  },
  "hdOtS-BpUPY_003550": {
    "video_start": 3550,
    "video_end": 3648,
    "anomaly_start": 49,
    "anomaly_end": 66,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 99,
    "subset": "val"
  },
  "hdOtS-BpUPY_003911": {
    "video_start": 3911,
    "video_end": 3979,
    "anomaly_start": 22,
    "anomaly_end": 31,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 69,
    "subset": "val"
  },
  "hdOtS-BpUPY_003981": {
    "video_start": 3981,
    "video_end": 4051,
    "anomaly_start": 29,
    "anomaly_end": 44,
    "anomaly_class": "other: start_stop_or_stationary",
    "num_frames": 71,
    "subset": "val"
  },
  "hdOtS-BpUPY_004503": {
    "video_start": 4503,
    "video_end": 4621,
    "anomaly_start": 28,
    "anomaly_end": 71,
    "anomaly_class": "other: start_stop_or_stationary",
    "num_frames": 119,
    "subset": "val"
  },
  "hdOtS-BpUPY_005541": {
    "video_start": 5541,
    "video_end": 5639,
    "anomaly_start": 52,
    "anomaly_end": 63,
    "anomaly_class": "ego: oncoming",
    "num_frames": 99,
    "subset": "val"
  },
  "hdOtS-BpUPY_005641": {
    "video_start": 5641,
    "video_end": 5739,
    "anomaly_start": 25,
    "anomaly_end": 64,
    "anomaly_class": "other: pedestrian",
    "num_frames": 99,
    "subset": "val"
  },
  "hfwb3mZZ0fs_000136": {
    "video_start": 136,
    "video_end": 241,
    "anomaly_start": 43,
    "anomaly_end": 75,
    "anomaly_class": "ego: lateral",
    "num_frames": 106,
    "subset": "val"
  },
  "hfwb3mZZ0fs_000243": {
    "video_start": 243,
    "video_end": 321,
    "anomaly_start": 34,
    "anomaly_end": 60,
    "anomaly_class": "ego: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "hfwb3mZZ0fs_000549": {
    "video_start": 549,
    "video_end": 667,
    "anomaly_start": 43,
    "anomaly_end": 96,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 119,
    "subset": "val"
  },
  "hfwb3mZZ0fs_000827": {
    "video_start": 827,
    "video_end": 971,
    "anomaly_start": 35,
    "anomaly_end": 115,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 145,
    "subset": "val"
  },
  "hfwb3mZZ0fs_000973": {
    "video_start": 973,
    "video_end": 1040,
    "anomaly_start": 33,
    "anomaly_end": 67,
    "anomaly_class": "ego: turning",
    "num_frames": 68,
    "subset": "val"
  },
  "hfwb3mZZ0fs_001042": {
    "video_start": 1042,
    "video_end": 1132,
    "anomaly_start": 25,
    "anomaly_end": 59,
    "anomaly_class": "ego: turning",
    "num_frames": 91,
    "subset": "val"
  },
  "hfwb3mZZ0fs_001747": {
    "video_start": 1747,
    "video_end": 1823,
    "anomaly_start": 17,
    "anomaly_end": 71,
    "anomaly_class": "other: lateral",
    "num_frames": 77,
    "subset": "val"
  },
  "hfwb3mZZ0fs_001977": {
    "video_start": 1977,
    "video_end": 2056,
    "anomaly_start": 26,
    "anomaly_end": 66,
    "anomaly_class": "ego: lateral",
    "num_frames": 80,
    "subset": "val"
  },
  "hfwb3mZZ0fs_002322": {
    "video_start": 2322,
    "video_end": 2464,
    "anomaly_start": 77,
    "anomaly_end": 119,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 143,
    "subset": "val"
  },
  "hfwb3mZZ0fs_003020": {
    "video_start": 3020,
    "video_end": 3107,
    "anomaly_start": 27,
    "anomaly_end": 75,
    "anomaly_class": "ego: turning",
    "num_frames": 88,
    "subset": "val"
  },
  "hfwb3mZZ0fs_004369": {
    "video_start": 4369,
    "video_end": 4574,
    "anomaly_start": 61,
    "anomaly_end": 174,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 206,
    "subset": "val"
  },
  "hy433gakFeo_001156": {
    "video_start": 1156,
    "video_end": 1249,
    "anomaly_start": 14,
    "anomaly_end": 42,
    "anomaly_class": "other: oncoming",
    "num_frames": 94,
    "subset": "val"
  },
  "i0Ya-cy2WgY_000635": {
    "video_start": 635,
    "video_end": 729,
    "anomaly_start": 14,
    "anomaly_end": 75,
    "anomaly_class": "ego: turning",
    "num_frames": 95,
    "subset": "val"
  },
  "ieP36TLihGM_000073": {
    "video_start": 73,
    "video_end": 189,
    "anomaly_start": 70,
    "anomaly_end": 96,
    "anomaly_class": "ego: turning",
    "num_frames": 117,
    "subset": "val"
  },
  "ieP36TLihGM_000874": {
    "video_start": 874,
    "video_end": 962,
    "anomaly_start": 52,
    "anomaly_end": 70,
    "anomaly_class": "ego: oncoming",
    "num_frames": 89,
    "subset": "val"
  },
  "ieP36TLihGM_000994": {
    "video_start": 994,
    "video_end": 1112,
    "anomaly_start": 33,
    "anomaly_end": 56,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 119,
    "subset": "val"
  },
  "ieP36TLihGM_001859": {
    "video_start": 1859,
    "video_end": 1950,
    "anomaly_start": 22,
    "anomaly_end": 72,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 92,
    "subset": "val"
  },
  "ieP36TLihGM_002282": {
    "video_start": 2282,
    "video_end": 2380,
    "anomaly_start": 20,
    "anomaly_end": 70,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 99,
    "subset": "val"
  },
  "ieP36TLihGM_002382": {
    "video_start": 2382,
    "video_end": 2571,
    "anomaly_start": 67,
    "anomaly_end": 107,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 190,
    "subset": "val"
  },
  "ieP36TLihGM_004004": {
    "video_start": 4004,
    "video_end": 4122,
    "anomaly_start": 51,
    "anomaly_end": 82,
    "anomaly_class": "ego: turning",
    "num_frames": 119,
    "subset": "val"
  },
  "ieP36TLihGM_004687": {
    "video_start": 4687,
    "video_end": 4795,
    "anomaly_start": 32,
    "anomaly_end": 75,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 109,
    "subset": "val"
  },
  "ieP36TLihGM_005277": {
    "video_start": 5277,
    "video_end": 5431,
    "anomaly_start": 53,
    "anomaly_end": 88,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 155,
    "subset": "val"
  },
  "jFFhYwgepmY_000073": {
    "video_start": 73,
    "video_end": 249,
    "anomaly_start": 81,
    "anomaly_end": 102,
    "anomaly_class": "ego: oncoming",
    "num_frames": 177,
    "subset": "val"
  },
  "jFFhYwgepmY_001008": {
    "video_start": 1008,
    "video_end": 1106,
    "anomaly_start": 44,
    "anomaly_end": 65,
    "anomaly_class": "ego: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "jFFhYwgepmY_001459": {
    "video_start": 1459,
    "video_end": 1577,
    "anomaly_start": 33,
    "anomaly_end": 78,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 119,
    "subset": "val"
  },
  "jFFhYwgepmY_001729": {
    "video_start": 1729,
    "video_end": 1856,
    "anomaly_start": 64,
    "anomaly_end": 73,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 128,
    "subset": "val"
  },
  "jFFhYwgepmY_003192": {
    "video_start": 3192,
    "video_end": 3330,
    "anomaly_start": 42,
    "anomaly_end": 76,
    "anomaly_class": "other: lateral",
    "num_frames": 139,
    "subset": "val"
  },
  "jFFhYwgepmY_003332": {
    "video_start": 3332,
    "video_end": 3440,
    "anomaly_start": 33,
    "anomaly_end": 54,
    "anomaly_class": "other: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "jFFhYwgepmY_003442": {
    "video_start": 3442,
    "video_end": 3560,
    "anomaly_start": 38,
    "anomaly_end": 94,
    "anomaly_class": "other: lateral",
    "num_frames": 119,
    "subset": "val"
  },
  "jFFhYwgepmY_004362": {
    "video_start": 4362,
    "video_end": 4460,
    "anomaly_start": 45,
    "anomaly_end": 62,
    "anomaly_class": "other: lateral",
    "num_frames": 99,
    "subset": "val"
  },
  "jFFhYwgepmY_004462": {
    "video_start": 4462,
    "video_end": 4570,
    "anomaly_start": 32,
    "anomaly_end": 76,
    "anomaly_class": "ego: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "jFFhYwgepmY_005022": {
    "video_start": 5022,
    "video_end": 5140,
    "anomaly_start": 56,
    "anomaly_end": 87,
    "anomaly_class": "ego: obstacle",
    "num_frames": 119,
    "subset": "val"
  },
  "jKwCC2RhxJE_001034": {
    "video_start": 1034,
    "video_end": 1140,
    "anomaly_start": 44,
    "anomaly_end": 92,
    "anomaly_class": "ego: lateral",
    "num_frames": 107,
    "subset": "val"
  },
  "jKwCC2RhxJE_002954": {
    "video_start": 2954,
    "video_end": 3026,
    "anomaly_start": 10,
    "anomaly_end": 60,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 73,
    "subset": "val"
  },
  "jv7vz_55H8M_000890": {
    "video_start": 890,
    "video_end": 985,
    "anomaly_start": 31,
    "anomaly_end": 69,
    "anomaly_class": "ego: turning",
    "num_frames": 96,
    "subset": "val"
  },
  "jv7vz_55H8M_000987": {
    "video_start": 987,
    "video_end": 1066,
    "anomaly_start": 39,
    "anomaly_end": 76,
    "anomaly_class": "ego: obstacle",
    "num_frames": 80,
    "subset": "val"
  },
  "jv7vz_55H8M_002578": {
    "video_start": 2578,
    "video_end": 2692,
    "anomaly_start": 27,
    "anomaly_end": 109,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 115,
    "subset": "val"
  },
  "kDo7FBuRiBs_001846": {
    "video_start": 1846,
    "video_end": 1934,
    "anomaly_start": 16,
    "anomaly_end": 75,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 89,
    "subset": "val"
  },
  "kN-2mc561GY_003223": {
    "video_start": 3223,
    "video_end": 3315,
    "anomaly_start": 57,
    "anomaly_end": 92,
    "anomaly_class": "ego: lateral",
    "num_frames": 93,
    "subset": "val"
  },
  "kN-2mc561GY_004683": {
    "video_start": 4683,
    "video_end": 4739,
    "anomaly_start": 12,
    "anomaly_end": 21,
    "anomaly_class": "ego: turning",
    "num_frames": 57,
    "subset": "val"
  },
  "kVhgFPPG-wo_001239": {
    "video_start": 1239,
    "video_end": 1318,
    "anomaly_start": 17,
    "anomaly_end": 50,
    "anomaly_class": "ego: turning",
    "num_frames": 80,
    "subset": "val"
  },
  "kVhgFPPG-wo_003102": {
    "video_start": 3102,
    "video_end": 3210,
    "anomaly_start": 24,
    "anomaly_end": 109,
    "anomaly_class": "ego: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "kVhgFPPG-wo_003212": {
    "video_start": 3212,
    "video_end": 3293,
    "anomaly_start": 20,
    "anomaly_end": 59,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 82,
    "subset": "val"
  },
  "kVhgFPPG-wo_004111": {
    "video_start": 4111,
    "video_end": 4183,
    "anomaly_start": 2,
    "anomaly_end": 53,
    "anomaly_class": "ego: lateral",
    "num_frames": 73,
    "subset": "val"
  },
  "kVhgFPPG-wo_004185": {
    "video_start": 4185,
    "video_end": 4280,
    "anomaly_start": 37,
    "anomaly_end": 72,
    "anomaly_class": "ego: lateral",
    "num_frames": 96,
    "subset": "val"
  },
  "kVhgFPPG-wo_004796": {
    "video_start": 4796,
    "video_end": 4906,
    "anomaly_start": 54,
    "anomaly_end": 101,
    "anomaly_class": "ego: lateral",
    "num_frames": 111,
    "subset": "val"
  },
  "kVhgFPPG-wo_005182": {
    "video_start": 5182,
    "video_end": 5276,
    "anomaly_start": 20,
    "anomaly_end": 61,
    "anomaly_class": "other: lateral",
    "num_frames": 95,
    "subset": "val"
  },
  "kVhgFPPG-wo_005860": {
    "video_start": 5860,
    "video_end": 5943,
    "anomaly_start": 21,
    "anomaly_end": 50,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 84,
    "subset": "val"
  },
  "kVhgFPPG-wo_006152": {
    "video_start": 6152,
    "video_end": 6270,
    "anomaly_start": 18,
    "anomaly_end": 119,
    "anomaly_class": "other: unknown",
    "num_frames": 119,
    "subset": "val"
  },
  "kjVywUi1WK4_000476": {
    "video_start": 476,
    "video_end": 572,
    "anomaly_start": 30,
    "anomaly_end": 64,
    "anomaly_class": "other: lateral",
    "num_frames": 97,
    "subset": "val"
  },
  "kjVywUi1WK4_001462": {
    "video_start": 1462,
    "video_end": 1568,
    "anomaly_start": 40,
    "anomaly_end": 68,
    "anomaly_class": "ego: turning",
    "num_frames": 107,
    "subset": "val"
  },
  "kjVywUi1WK4_001658": {
    "video_start": 1658,
    "video_end": 1724,
    "anomaly_start": 25,
    "anomaly_end": 45,
    "anomaly_class": "other: start_stop_or_stationary",
    "num_frames": 67,
    "subset": "val"
  },
  "kjVywUi1WK4_001814": {
    "video_start": 1814,
    "video_end": 1925,
    "anomaly_start": 38,
    "anomaly_end": 112,
    "anomaly_class": "ego: oncoming",
    "num_frames": 112,
    "subset": "val"
  },
  "kjVywUi1WK4_002026": {
    "video_start": 2026,
    "video_end": 2121,
    "anomaly_start": 44,
    "anomaly_end": 66,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 96,
    "subset": "val"
  },
  "kjVywUi1WK4_003685": {
    "video_start": 3685,
    "video_end": 3759,
    "anomaly_start": 48,
    "anomaly_end": 74,
    "anomaly_class": "ego: lateral",
    "num_frames": 75,
    "subset": "val"
  },
  "kjVywUi1WK4_004128": {
    "video_start": 4128,
    "video_end": 4245,
    "anomaly_start": 78,
    "anomaly_end": 106,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 118,
    "subset": "val"
  },
  "kld-iYpdjyo_000786": {
    "video_start": 786,
    "video_end": 873,
    "anomaly_start": 22,
    "anomaly_end": 47,
    "anomaly_class": "other: unknown",
    "num_frames": 88,
    "subset": "val"
  },
  "kld-iYpdjyo_002031": {
    "video_start": 2031,
    "video_end": 2129,
    "anomaly_start": 16,
    "anomaly_end": 62,
    "anomaly_class": "other: lateral",
    "num_frames": 99,
    "subset": "val"
  },
  "kld-iYpdjyo_002713": {
    "video_start": 2713,
    "video_end": 2833,
    "anomaly_start": 37,
    "anomaly_end": 80,
    "anomaly_class": "other: lateral",
    "num_frames": 121,
    "subset": "val"
  },
  "kld-iYpdjyo_004278": {
    "video_start": 4278,
    "video_end": 4407,
    "anomaly_start": 35,
    "anomaly_end": 61,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 130,
    "subset": "val"
  },
  "kld-iYpdjyo_004761": {
    "video_start": 4761,
    "video_end": 4839,
    "anomaly_start": 27,
    "anomaly_end": 48,
    "anomaly_class": "other: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "kld-iYpdjyo_004979": {
    "video_start": 4979,
    "video_end": 5064,
    "anomaly_start": 31,
    "anomaly_end": 58,
    "anomaly_class": "ego: lateral",
    "num_frames": 86,
    "subset": "val"
  },
  "kld-iYpdjyo_005765": {
    "video_start": 5765,
    "video_end": 5913,
    "anomaly_start": 57,
    "anomaly_end": 97,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 149,
    "subset": "val"
  },
  "kld-iYpdjyo_006125": {
    "video_start": 6125,
    "video_end": 6186,
    "anomaly_start": 28,
    "anomaly_end": 53,
    "anomaly_class": "other: pedestrian",
    "num_frames": 62,
    "subset": "val"
  },
  "krPRThe7R4g_000719": {
    "video_start": 719,
    "video_end": 794,
    "anomaly_start": 31,
    "anomaly_end": 57,
    "anomaly_class": "other: turning",
    "num_frames": 76,
    "subset": "val"
  },
  "krPRThe7R4g_004374": {
    "video_start": 4374,
    "video_end": 4447,
    "anomaly_start": 17,
    "anomaly_end": 35,
    "anomaly_class": "ego: turning",
    "num_frames": 74,
    "subset": "val"
  },
  "l1XbmdmGr0Q_000274": {
    "video_start": 274,
    "video_end": 372,
    "anomaly_start": 38,
    "anomaly_end": 80,
    "anomaly_class": "ego: lateral",
    "num_frames": 99,
    "subset": "val"
  },
  "l1XbmdmGr0Q_000374": {
    "video_start": 374,
    "video_end": 448,
    "anomaly_start": 40,
    "anomaly_end": 52,
    "anomaly_class": "ego: lateral",
    "num_frames": 75,
    "subset": "val"
  },
  "l1XbmdmGr0Q_000857": {
    "video_start": 857,
    "video_end": 975,
    "anomaly_start": 26,
    "anomaly_end": 43,
    "anomaly_class": "ego: oncoming",
    "num_frames": 119,
    "subset": "val"
  },
  "l1XbmdmGr0Q_001914": {
    "video_start": 1914,
    "video_end": 2003,
    "anomaly_start": 44,
    "anomaly_end": 69,
    "anomaly_class": "other: turning",
    "num_frames": 90,
    "subset": "val"
  },
  "l1XbmdmGr0Q_002005": {
    "video_start": 2005,
    "video_end": 2083,
    "anomaly_start": 25,
    "anomaly_end": 56,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 79,
    "subset": "val"
  },
  "l1XbmdmGr0Q_002358": {
    "video_start": 2358,
    "video_end": 2458,
    "anomaly_start": 54,
    "anomaly_end": 85,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 101,
    "subset": "val"
  },
  "l1XbmdmGr0Q_003081": {
    "video_start": 3081,
    "video_end": 3168,
    "anomaly_start": 36,
    "anomaly_end": 64,
    "anomaly_class": "ego: lateral",
    "num_frames": 88,
    "subset": "val"
  },
  "l1XbmdmGr0Q_004518": {
    "video_start": 4518,
    "video_end": 4632,
    "anomaly_start": 35,
    "anomaly_end": 59,
    "anomaly_class": "other: lateral",
    "num_frames": 115,
    "subset": "val"
  },
  "l1XbmdmGr0Q_005056": {
    "video_start": 5056,
    "video_end": 5118,
    "anomaly_start": 16,
    "anomaly_end": 42,
    "anomaly_class": "ego: turning",
    "num_frames": 63,
    "subset": "val"
  },
  "lKMwX4nA64k_000052": {
    "video_start": 52,
    "video_end": 146,
    "anomaly_start": 45,
    "anomaly_end": 58,
    "anomaly_class": "ego: oncoming",
    "num_frames": 95,
    "subset": "val"
  },
  "lKMwX4nA64k_000474": {
    "video_start": 474,
    "video_end": 574,
    "anomaly_start": 36,
    "anomaly_end": 55,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 101,
    "subset": "val"
  },
  "lKMwX4nA64k_001027": {
    "video_start": 1027,
    "video_end": 1137,
    "anomaly_start": 16,
    "anomaly_end": 35,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 111,
    "subset": "val"
  },
  "lKMwX4nA64k_001689": {
    "video_start": 1689,
    "video_end": 1780,
    "anomaly_start": 43,
    "anomaly_end": 62,
    "anomaly_class": "other: pedestrian",
    "num_frames": 92,
    "subset": "val"
  },
  "lKMwX4nA64k_001782": {
    "video_start": 1782,
    "video_end": 1924,
    "anomaly_start": 7,
    "anomaly_end": 80,
    "anomaly_class": "other: lateral",
    "num_frames": 143,
    "subset": "val"
  },
  "lKMwX4nA64k_002645": {
    "video_start": 2645,
    "video_end": 2734,
    "anomaly_start": 50,
    "anomaly_end": 70,
    "anomaly_class": "other: turning",
    "num_frames": 90,
    "subset": "val"
  },
  "lKMwX4nA64k_003285": {
    "video_start": 3285,
    "video_end": 3386,
    "anomaly_start": 49,
    "anomaly_end": 73,
    "anomaly_class": "ego: lateral",
    "num_frames": 102,
    "subset": "val"
  },
  "lKMwX4nA64k_003568": {
    "video_start": 3568,
    "video_end": 3654,
    "anomaly_start": 28,
    "anomaly_end": 54,
    "anomaly_class": "ego: turning",
    "num_frames": 87,
    "subset": "val"
  },
  "lKMwX4nA64k_003913": {
    "video_start": 3913,
    "video_end": 4012,
    "anomaly_start": 22,
    "anomaly_end": 43,
    "anomaly_class": "other: turning",
    "num_frames": 100,
    "subset": "val"
  },
  "lKMwX4nA64k_004789": {
    "video_start": 4789,
    "video_end": 4886,
    "anomaly_start": 38,
    "anomaly_end": 68,
    "anomaly_class": "ego: lateral",
    "num_frames": 98,
    "subset": "val"
  },
  "lKMwX4nA64k_005351": {
    "video_start": 5351,
    "video_end": 5417,
    "anomaly_start": 19,
    "anomaly_end": 46,
    "anomaly_class": "ego: turning",
    "num_frames": 67,
    "subset": "val"
  },
  "lmykriTbncM_002210": {
    "video_start": 2210,
    "video_end": 2290,
    "anomaly_start": 24,
    "anomaly_end": 45,
    "anomaly_class": "other: start_stop_or_stationary",
    "num_frames": 81,
    "subset": "val"
  },
  "lmykriTbncM_002796": {
    "video_start": 2796,
    "video_end": 2884,
    "anomaly_start": 8,
    "anomaly_end": 31,
    "anomaly_class": "other: lateral",
    "num_frames": 89,
    "subset": "val"
  },
  "lmykriTbncM_002886": {
    "video_start": 2886,
    "video_end": 2998,
    "anomaly_start": 46,
    "anomaly_end": 64,
    "anomaly_class": "ego: oncoming",
    "num_frames": 113,
    "subset": "val"
  },
  "lmykriTbncM_003292": {
    "video_start": 3292,
    "video_end": 3405,
    "anomaly_start": 46,
    "anomaly_end": 69,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 114,
    "subset": "val"
  },
  "lmykriTbncM_003990": {
    "video_start": 3990,
    "video_end": 4070,
    "anomaly_start": 4,
    "anomaly_end": 48,
    "anomaly_class": "ego: lateral",
    "num_frames": 81,
    "subset": "val"
  },
  "lmykriTbncM_004072": {
    "video_start": 4072,
    "video_end": 4160,
    "anomaly_start": 50,
    "anomaly_end": 68,
    "anomaly_class": "other: lateral",
    "num_frames": 89,
    "subset": "val"
  },
  "mrIFkGZbNjc_000652": {
    "video_start": 652,
    "video_end": 750,
    "anomaly_start": 43,
    "anomaly_end": 54,
    "anomaly_class": "other: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "mrIFkGZbNjc_000953": {
    "video_start": 953,
    "video_end": 1021,
    "anomaly_start": 15,
    "anomaly_end": 40,
    "anomaly_class": "other: turning",
    "num_frames": 69,
    "subset": "val"
  },
  "mrIFkGZbNjc_001023": {
    "video_start": 1023,
    "video_end": 1131,
    "anomaly_start": 48,
    "anomaly_end": 62,
    "anomaly_class": "other: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "mrIFkGZbNjc_001416": {
    "video_start": 1416,
    "video_end": 1554,
    "anomaly_start": 28,
    "anomaly_end": 94,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 139,
    "subset": "val"
  },
  "mrIFkGZbNjc_003436": {
    "video_start": 3436,
    "video_end": 3575,
    "anomaly_start": 48,
    "anomaly_end": 70,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 140,
    "subset": "val"
  },
  "nADqn-DZ-Dc_000075": {
    "video_start": 75,
    "video_end": 209,
    "anomaly_start": 44,
    "anomaly_end": 121,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 135,
    "subset": "val"
  },
  "nADqn-DZ-Dc_000430": {
    "video_start": 430,
    "video_end": 528,
    "anomaly_start": 53,
    "anomaly_end": 73,
    "anomaly_class": "other: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "nADqn-DZ-Dc_000530": {
    "video_start": 530,
    "video_end": 638,
    "anomaly_start": 52,
    "anomaly_end": 81,
    "anomaly_class": "other: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "nADqn-DZ-Dc_000850": {
    "video_start": 850,
    "video_end": 988,
    "anomaly_start": 82,
    "anomaly_end": 96,
    "anomaly_class": "other: turning",
    "num_frames": 139,
    "subset": "val"
  },
  "nHvTqyRntbY_000914": {
    "video_start": 914,
    "video_end": 1033,
    "anomaly_start": 42,
    "anomaly_end": 63,
    "anomaly_class": "ego: unknown",
    "num_frames": 120,
    "subset": "val"
  },
  "nHvTqyRntbY_001259": {
    "video_start": 1259,
    "video_end": 1339,
    "anomaly_start": 14,
    "anomaly_end": 25,
    "anomaly_class": "other: turning",
    "num_frames": 81,
    "subset": "val"
  },
  "nHvTqyRntbY_001534": {
    "video_start": 1534,
    "video_end": 1602,
    "anomaly_start": 25,
    "anomaly_end": 43,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 69,
    "subset": "val"
  },
  "nHvTqyRntbY_001604": {
    "video_start": 1604,
    "video_end": 1695,
    "anomaly_start": 31,
    "anomaly_end": 45,
    "anomaly_class": "other: obstacle",
    "num_frames": 92,
    "subset": "val"
  },
  "nHvTqyRntbY_003485": {
    "video_start": 3485,
    "video_end": 3565,
    "anomaly_start": 34,
    "anomaly_end": 51,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 81,
    "subset": "val"
  },
  "nHvTqyRntbY_003567": {
    "video_start": 3567,
    "video_end": 3655,
    "anomaly_start": 30,
    "anomaly_end": 51,
    "anomaly_class": "other: lateral",
    "num_frames": 89,
    "subset": "val"
  },
  "nvVk7L3GrJQ_000051": {
    "video_start": 51,
    "video_end": 179,
    "anomaly_start": 43,
    "anomaly_end": 81,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 129,
    "subset": "val"
  },
  "nvVk7L3GrJQ_000181": {
    "video_start": 181,
    "video_end": 335,
    "anomaly_start": 45,
    "anomaly_end": 93,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 155,
    "subset": "val"
  },
  "nvVk7L3GrJQ_000856": {
    "video_start": 856,
    "video_end": 999,
    "anomaly_start": 61,
    "anomaly_end": 142,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 144,
    "subset": "val"
  },
  "nvVk7L3GrJQ_001001": {
    "video_start": 1001,
    "video_end": 1080,
    "anomaly_start": 27,
    "anomaly_end": 40,
    "anomaly_class": "other: start_stop_or_stationary",
    "num_frames": 80,
    "subset": "val"
  },
  "nvVk7L3GrJQ_001382": {
    "video_start": 1382,
    "video_end": 1456,
    "anomaly_start": 28,
    "anomaly_end": 43,
    "anomaly_class": "ego: turning",
    "num_frames": 75,
    "subset": "val"
  },
  "nvVk7L3GrJQ_002404": {
    "video_start": 2404,
    "video_end": 2529,
    "anomaly_start": 31,
    "anomaly_end": 76,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 126,
    "subset": "val"
  },
  "nvVk7L3GrJQ_002531": {
    "video_start": 2531,
    "video_end": 2648,
    "anomaly_start": 42,
    "anomaly_end": 62,
    "anomaly_class": "ego: turning",
    "num_frames": 118,
    "subset": "val"
  },
  "nvVk7L3GrJQ_002895": {
    "video_start": 2895,
    "video_end": 2977,
    "anomaly_start": 21,
    "anomaly_end": 40,
    "anomaly_class": "other: turning",
    "num_frames": 83,
    "subset": "val"
  },
  "nvVk7L3GrJQ_003279": {
    "video_start": 3279,
    "video_end": 3347,
    "anomaly_start": 42,
    "anomaly_end": 69,
    "anomaly_class": "ego: oncoming",
    "num_frames": 69,
    "subset": "val"
  },
  "nvVk7L3GrJQ_003857": {
    "video_start": 3857,
    "video_end": 3928,
    "anomaly_start": 28,
    "anomaly_end": 52,
    "anomaly_class": "ego: obstacle",
    "num_frames": 72,
    "subset": "val"
  },
  "nvVk7L3GrJQ_004517": {
    "video_start": 4517,
    "video_end": 4588,
    "anomaly_start": 35,
    "anomaly_end": 52,
    "anomaly_class": "ego: turning",
    "num_frames": 72,
    "subset": "val"
  },
  "nvVk7L3GrJQ_004707": {
    "video_start": 4707,
    "video_end": 4797,
    "anomaly_start": 57,
    "anomaly_end": 90,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 91,
    "subset": "val"
  },
  "oZaUkhs_H-s_000501": {
    "video_start": 501,
    "video_end": 609,
    "anomaly_start": 45,
    "anomaly_end": 67,
    "anomaly_class": "ego: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "oZaUkhs_H-s_000611": {
    "video_start": 611,
    "video_end": 719,
    "anomaly_start": 42,
    "anomaly_end": 82,
    "anomaly_class": "other: oncoming",
    "num_frames": 109,
    "subset": "val"
  },
  "oZaUkhs_H-s_000927": {
    "video_start": 927,
    "video_end": 1035,
    "anomaly_start": 31,
    "anomaly_end": 78,
    "anomaly_class": "ego: oncoming",
    "num_frames": 109,
    "subset": "val"
  },
  "oZaUkhs_H-s_003307": {
    "video_start": 3307,
    "video_end": 3375,
    "anomaly_start": 30,
    "anomaly_end": 48,
    "anomaly_class": "other: turning",
    "num_frames": 69,
    "subset": "val"
  },
  "oZaUkhs_H-s_004282": {
    "video_start": 4282,
    "video_end": 4360,
    "anomaly_start": 38,
    "anomaly_end": 60,
    "anomaly_class": "ego: oncoming",
    "num_frames": 79,
    "subset": "val"
  },
  "oZaUkhs_H-s_004563": {
    "video_start": 4563,
    "video_end": 4701,
    "anomaly_start": 41,
    "anomaly_end": 127,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 139,
    "subset": "val"
  },
  "oZaUkhs_H-s_005205": {
    "video_start": 5205,
    "video_end": 5303,
    "anomaly_start": 41,
    "anomaly_end": 71,
    "anomaly_class": "other: obstacle",
    "num_frames": 99,
    "subset": "val"
  },
  "p-fBcE77G4c_000692": {
    "video_start": 692,
    "video_end": 813,
    "anomaly_start": 47,
    "anomaly_end": 90,
    "anomaly_class": "other: lateral",
    "num_frames": 122,
    "subset": "val"
  },
  "p-fBcE77G4c_001420": {
    "video_start": 1420,
    "video_end": 1504,
    "anomaly_start": 18,
    "anomaly_end": 30,
    "anomaly_class": "ego: turning",
    "num_frames": 85,
    "subset": "val"
  },
  "p-fBcE77G4c_001746": {
    "video_start": 1746,
    "video_end": 1824,
    "anomaly_start": 40,
    "anomaly_end": 60,
    "anomaly_class": "ego: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "p-fBcE77G4c_003212": {
    "video_start": 3212,
    "video_end": 3309,
    "anomaly_start": 29,
    "anomaly_end": 51,
    "anomaly_class": "ego: turning",
    "num_frames": 98,
    "subset": "val"
  },
  "p-fBcE77G4c_003857": {
    "video_start": 3857,
    "video_end": 3928,
    "anomaly_start": 36,
    "anomaly_end": 52,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 72,
    "subset": "val"
  },
  "p-fBcE77G4c_004839": {
    "video_start": 4839,
    "video_end": 4895,
    "anomaly_start": 10,
    "anomaly_end": 31,
    "anomaly_class": "other: turning",
    "num_frames": 57,
    "subset": "val"
  },
  "p-fBcE77G4c_004988": {
    "video_start": 4988,
    "video_end": 5052,
    "anomaly_start": 23,
    "anomaly_end": 40,
    "anomaly_class": "other: start_stop_or_stationary",
    "num_frames": 65,
    "subset": "val"
  },
  "p-fBcE77G4c_005054": {
    "video_start": 5054,
    "video_end": 5115,
    "anomaly_start": 24,
    "anomaly_end": 33,
    "anomaly_class": "ego: lateral",
    "num_frames": 62,
    "subset": "val"
  },
  "p8q77QzOdUs_000652": {
    "video_start": 652,
    "video_end": 730,
    "anomaly_start": 33,
    "anomaly_end": 51,
    "anomaly_class": "ego: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "p8q77QzOdUs_005240": {
    "video_start": 5240,
    "video_end": 5348,
    "anomaly_start": 51,
    "anomaly_end": 73,
    "anomaly_class": "other: start_stop_or_stationary",
    "num_frames": 109,
    "subset": "val"
  },
  "pOJpwWmEcFg_000075": {
    "video_start": 75,
    "video_end": 189,
    "anomaly_start": 55,
    "anomaly_end": 80,
    "anomaly_class": "other: turning",
    "num_frames": 115,
    "subset": "val"
  },
  "pOJpwWmEcFg_000191": {
    "video_start": 191,
    "video_end": 279,
    "anomaly_start": 46,
    "anomaly_end": 61,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "pOJpwWmEcFg_001615": {
    "video_start": 1615,
    "video_end": 1726,
    "anomaly_start": 49,
    "anomaly_end": 78,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 112,
    "subset": "val"
  },
  "pOJpwWmEcFg_005127": {
    "video_start": 5127,
    "video_end": 5200,
    "anomaly_start": 20,
    "anomaly_end": 47,
    "anomaly_class": "ego: turning",
    "num_frames": 74,
    "subset": "val"
  },
  "pOJpwWmEcFg_005342": {
    "video_start": 5342,
    "video_end": 5448,
    "anomaly_start": 38,
    "anomaly_end": 54,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 107,
    "subset": "val"
  },
  "pQdl0apLT70_000061": {
    "video_start": 61,
    "video_end": 171,
    "anomaly_start": 17,
    "anomaly_end": 37,
    "anomaly_class": "other: turning",
    "num_frames": 111,
    "subset": "val"
  },
  "pQdl0apLT70_000883": {
    "video_start": 883,
    "video_end": 945,
    "anomaly_start": 35,
    "anomaly_end": 58,
    "anomaly_class": "ego: turning",
    "num_frames": 63,
    "subset": "val"
  },
  "pQdl0apLT70_001303": {
    "video_start": 1303,
    "video_end": 1372,
    "anomaly_start": 19,
    "anomaly_end": 43,
    "anomaly_class": "other: lateral",
    "num_frames": 70,
    "subset": "val"
  },
  "pQdl0apLT70_002193": {
    "video_start": 2193,
    "video_end": 2283,
    "anomaly_start": 44,
    "anomaly_end": 62,
    "anomaly_class": "other: pedestrian",
    "num_frames": 91,
    "subset": "val"
  },
  "pQdl0apLT70_002720": {
    "video_start": 2720,
    "video_end": 2817,
    "anomaly_start": 28,
    "anomaly_end": 48,
    "anomaly_class": "ego: turning",
    "num_frames": 98,
    "subset": "val"
  },
  "pQdl0apLT70_003045": {
    "video_start": 3045,
    "video_end": 3131,
    "anomaly_start": 37,
    "anomaly_end": 50,
    "anomaly_class": "ego: lateral",
    "num_frames": 87,
    "subset": "val"
  },
  "pQdl0apLT70_004341": {
    "video_start": 4341,
    "video_end": 4450,
    "anomaly_start": 32,
    "anomaly_end": 83,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 110,
    "subset": "val"
  },
  "pQdl0apLT70_004600": {
    "video_start": 4600,
    "video_end": 4680,
    "anomaly_start": 33,
    "anomaly_end": 52,
    "anomaly_class": "ego: oncoming",
    "num_frames": 81,
    "subset": "val"
  },
  "pQdl0apLT70_004976": {
    "video_start": 4976,
    "video_end": 5066,
    "anomaly_start": 40,
    "anomaly_end": 76,
    "anomaly_class": "ego: oncoming",
    "num_frames": 91,
    "subset": "val"
  },
  "pWZSG2cUdEo_000392": {
    "video_start": 392,
    "video_end": 463,
    "anomaly_start": 16,
    "anomaly_end": 38,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 72,
    "subset": "val"
  },
  "pWZSG2cUdEo_000597": {
    "video_start": 597,
    "video_end": 718,
    "anomaly_start": 57,
    "anomaly_end": 99,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 122,
    "subset": "val"
  },
  "pWZSG2cUdEo_001061": {
    "video_start": 1061,
    "video_end": 1137,
    "anomaly_start": 28,
    "anomaly_end": 41,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 77,
    "subset": "val"
  },
  "pWZSG2cUdEo_001422": {
    "video_start": 1422,
    "video_end": 1518,
    "anomaly_start": 39,
    "anomaly_end": 49,
    "anomaly_class": "ego: pedestrian",
    "num_frames": 97,
    "subset": "val"
  },
  "pWZSG2cUdEo_001753": {
    "video_start": 1753,
    "video_end": 1887,
    "anomaly_start": 60,
    "anomaly_end": 93,
    "anomaly_class": "ego: obstacle",
    "num_frames": 135,
    "subset": "val"
  },
  "pWZSG2cUdEo_002536": {
    "video_start": 2536,
    "video_end": 2625,
    "anomaly_start": 44,
    "anomaly_end": 77,
    "anomaly_class": "other: lateral",
    "num_frames": 90,
    "subset": "val"
  },
  "pWZSG2cUdEo_002627": {
    "video_start": 2627,
    "video_end": 2828,
    "anomaly_start": 48,
    "anomaly_end": 75,
    "anomaly_class": "ego: lateral",
    "num_frames": 202,
    "subset": "val"
  },
  "pWZSG2cUdEo_003185": {
    "video_start": 3185,
    "video_end": 3285,
    "anomaly_start": 28,
    "anomaly_end": 69,
    "anomaly_class": "ego: turning",
    "num_frames": 101,
    "subset": "val"
  },
  "pWZSG2cUdEo_003288": {
    "video_start": 3288,
    "video_end": 3361,
    "anomaly_start": 34,
    "anomaly_end": 48,
    "anomaly_class": "ego: turning",
    "num_frames": 74,
    "subset": "val"
  },
  "pWZSG2cUdEo_003625": {
    "video_start": 3625,
    "video_end": 3795,
    "anomaly_start": 19,
    "anomaly_end": 27,
    "anomaly_class": "other: lateral",
    "num_frames": 171,
    "subset": "val"
  },
  "pWZSG2cUdEo_003797": {
    "video_start": 3797,
    "video_end": 3822,
    "anomaly_start": 19,
    "anomaly_end": 26,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 26,
    "subset": "val"
  },
  "qYonJ4w6SZs_000191": {
    "video_start": 191,
    "video_end": 330,
    "anomaly_start": 76,
    "anomaly_end": 108,
    "anomaly_class": "ego: turning",
    "num_frames": 140,
    "subset": "val"
  },
  "qYonJ4w6SZs_000332": {
    "video_start": 332,
    "video_end": 427,
    "anomaly_start": 21,
    "anomaly_end": 62,
    "anomaly_class": "ego: turning",
    "num_frames": 96,
    "subset": "val"
  },
  "qYonJ4w6SZs_002152": {
    "video_start": 2152,
    "video_end": 2251,
    "anomaly_start": 38,
    "anomaly_end": 78,
    "anomaly_class": "ego: unknown",
    "num_frames": 100,
    "subset": "val"
  },
  "qYonJ4w6SZs_002253": {
    "video_start": 2253,
    "video_end": 2391,
    "anomaly_start": 43,
    "anomaly_end": 84,
    "anomaly_class": "ego: lateral",
    "num_frames": 139,
    "subset": "val"
  },
  "qYonJ4w6SZs_002805": {
    "video_start": 2805,
    "video_end": 2944,
    "anomaly_start": 60,
    "anomaly_end": 83,
    "anomaly_class": "other: turning",
    "num_frames": 140,
    "subset": "val"
  },
  "qYonJ4w6SZs_003336": {
    "video_start": 3336,
    "video_end": 3452,
    "anomaly_start": 64,
    "anomaly_end": 76,
    "anomaly_class": "ego: oncoming",
    "num_frames": 117,
    "subset": "val"
  },
  "qYonJ4w6SZs_003690": {
    "video_start": 3690,
    "video_end": 3778,
    "anomaly_start": 31,
    "anomaly_end": 69,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "qYonJ4w6SZs_003780": {
    "video_start": 3780,
    "video_end": 3870,
    "anomaly_start": 33,
    "anomaly_end": 62,
    "anomaly_class": "other: turning",
    "num_frames": 91,
    "subset": "val"
  },
  "qYonJ4w6SZs_004844": {
    "video_start": 4844,
    "video_end": 4932,
    "anomaly_start": 39,
    "anomaly_end": 62,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "qYonJ4w6SZs_005160": {
    "video_start": 5160,
    "video_end": 5258,
    "anomaly_start": 36,
    "anomaly_end": 61,
    "anomaly_class": "ego: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "qYonJ4w6SZs_005724": {
    "video_start": 5724,
    "video_end": 5841,
    "anomaly_start": 55,
    "anomaly_end": 96,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 118,
    "subset": "val"
  },
  "qzMjfBx1KI0_000603": {
    "video_start": 603,
    "video_end": 714,
    "anomaly_start": 35,
    "anomaly_end": 56,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 112,
    "subset": "val"
  },
  "qzMjfBx1KI0_002475": {
    "video_start": 2475,
    "video_end": 2533,
    "anomaly_start": 25,
    "anomaly_end": 46,
    "anomaly_class": "ego: turning",
    "num_frames": 59,
    "subset": "val"
  },
  "qzMjfBx1KI0_002785": {
    "video_start": 2785,
    "video_end": 2868,
    "anomaly_start": 22,
    "anomaly_end": 47,
    "anomaly_class": "ego: lateral",
    "num_frames": 84,
    "subset": "val"
  },
  "qzMjfBx1KI0_003085": {
    "video_start": 3085,
    "video_end": 3178,
    "anomaly_start": 27,
    "anomaly_end": 79,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 94,
    "subset": "val"
  },
  "qzMjfBx1KI0_004529": {
    "video_start": 4529,
    "video_end": 4629,
    "anomaly_start": 31,
    "anomaly_end": 41,
    "anomaly_class": "ego: turning",
    "num_frames": 101,
    "subset": "val"
  },
  "qzMjfBx1KI0_005445": {
    "video_start": 5445,
    "video_end": 5496,
    "anomaly_start": 21,
    "anomaly_end": 43,
    "anomaly_class": "ego: obstacle",
    "num_frames": 52,
    "subset": "val"
  },
  "r6_ZhT7rmhM_000324": {
    "video_start": 324,
    "video_end": 434,
    "anomaly_start": 54,
    "anomaly_end": 77,
    "anomaly_class": "ego: turning",
    "num_frames": 111,
    "subset": "val"
  },
  "r6_ZhT7rmhM_000644": {
    "video_start": 644,
    "video_end": 708,
    "anomaly_start": 24,
    "anomaly_end": 33,
    "anomaly_class": "ego: turning",
    "num_frames": 65,
    "subset": "val"
  },
  "r6_ZhT7rmhM_001168": {
    "video_start": 1168,
    "video_end": 1256,
    "anomaly_start": 34,
    "anomaly_end": 44,
    "anomaly_class": "ego: lateral",
    "num_frames": 89,
    "subset": "val"
  },
  "r6_ZhT7rmhM_001434": {
    "video_start": 1434,
    "video_end": 1521,
    "anomaly_start": 31,
    "anomaly_end": 60,
    "anomaly_class": "ego: turning",
    "num_frames": 88,
    "subset": "val"
  },
  "r6_ZhT7rmhM_001523": {
    "video_start": 1523,
    "video_end": 1614,
    "anomaly_start": 40,
    "anomaly_end": 67,
    "anomaly_class": "other: lateral",
    "num_frames": 92,
    "subset": "val"
  },
  "r6_ZhT7rmhM_001724": {
    "video_start": 1724,
    "video_end": 1795,
    "anomaly_start": 18,
    "anomaly_end": 35,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 72,
    "subset": "val"
  },
  "r6_ZhT7rmhM_002869": {
    "video_start": 2869,
    "video_end": 2949,
    "anomaly_start": 50,
    "anomaly_end": 81,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 81,
    "subset": "val"
  },
  "r6_ZhT7rmhM_004507": {
    "video_start": 4507,
    "video_end": 4613,
    "anomaly_start": 54,
    "anomaly_end": 89,
    "anomaly_class": "other: turning",
    "num_frames": 107,
    "subset": "val"
  },
  "r6_ZhT7rmhM_004807": {
    "video_start": 4807,
    "video_end": 4890,
    "anomaly_start": 29,
    "anomaly_end": 44,
    "anomaly_class": "other: turning",
    "num_frames": 84,
    "subset": "val"
  },
  "ruiSuGHJf6Q_000075": {
    "video_start": 75,
    "video_end": 189,
    "anomaly_start": 59,
    "anomaly_end": 96,
    "anomaly_class": "ego: turning",
    "num_frames": 115,
    "subset": "val"
  },
  "ruiSuGHJf6Q_000191": {
    "video_start": 191,
    "video_end": 269,
    "anomaly_start": 42,
    "anomaly_end": 66,
    "anomaly_class": "other: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "ruiSuGHJf6Q_000271": {
    "video_start": 271,
    "video_end": 369,
    "anomaly_start": 54,
    "anomaly_end": 73,
    "anomaly_class": "ego: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "ruiSuGHJf6Q_000771": {
    "video_start": 771,
    "video_end": 879,
    "anomaly_start": 38,
    "anomaly_end": 71,
    "anomaly_class": "other: obstacle",
    "num_frames": 109,
    "subset": "val"
  },
  "ruiSuGHJf6Q_001131": {
    "video_start": 1131,
    "video_end": 1229,
    "anomaly_start": 42,
    "anomaly_end": 69,
    "anomaly_class": "other: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "ruiSuGHJf6Q_001501": {
    "video_start": 1501,
    "video_end": 1639,
    "anomaly_start": 85,
    "anomaly_end": 109,
    "anomaly_class": "ego: turning",
    "num_frames": 139,
    "subset": "val"
  },
  "ruiSuGHJf6Q_003009": {
    "video_start": 3009,
    "video_end": 3217,
    "anomaly_start": 44,
    "anomaly_end": 155,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 209,
    "subset": "val"
  },
  "ruiSuGHJf6Q_004479": {
    "video_start": 4479,
    "video_end": 4558,
    "anomaly_start": 44,
    "anomaly_end": 60,
    "anomaly_class": "other: oncoming",
    "num_frames": 80,
    "subset": "val"
  },
  "ruiSuGHJf6Q_005729": {
    "video_start": 5729,
    "video_end": 5877,
    "anomaly_start": 108,
    "anomaly_end": 126,
    "anomaly_class": "ego: lateral",
    "num_frames": 149,
    "subset": "val"
  },
  "sUbP4j8rAic_000075": {
    "video_start": 75,
    "video_end": 170,
    "anomaly_start": 47,
    "anomaly_end": 74,
    "anomaly_class": "ego: turning",
    "num_frames": 96,
    "subset": "val"
  },
  "sUbP4j8rAic_000820": {
    "video_start": 820,
    "video_end": 928,
    "anomaly_start": 34,
    "anomaly_end": 64,
    "anomaly_class": "other: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "sUbP4j8rAic_001450": {
    "video_start": 1450,
    "video_end": 1538,
    "anomaly_start": 42,
    "anomaly_end": 61,
    "anomaly_class": "other: start_stop_or_stationary",
    "num_frames": 89,
    "subset": "val"
  },
  "sUbP4j8rAic_001540": {
    "video_start": 1540,
    "video_end": 1691,
    "anomaly_start": 43,
    "anomaly_end": 111,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 152,
    "subset": "val"
  },
  "sUbP4j8rAic_002038": {
    "video_start": 2038,
    "video_end": 2138,
    "anomaly_start": 39,
    "anomaly_end": 65,
    "anomaly_class": "other: oncoming",
    "num_frames": 101,
    "subset": "val"
  },
  "sUbP4j8rAic_003555": {
    "video_start": 3555,
    "video_end": 3723,
    "anomaly_start": 18,
    "anomaly_end": 31,
    "anomaly_class": "other: start_stop_or_stationary",
    "num_frames": 169,
    "subset": "val"
  },
  "sUbP4j8rAic_003725": {
    "video_start": 3725,
    "video_end": 3777,
    "anomaly_start": 43,
    "anomaly_end": 53,
    "anomaly_class": "ego: oncoming",
    "num_frames": 53,
    "subset": "val"
  },
  "sUbP4j8rAic_004739": {
    "video_start": 4739,
    "video_end": 4827,
    "anomaly_start": 33,
    "anomaly_end": 60,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "sUbP4j8rAic_005029": {
    "video_start": 5029,
    "video_end": 5124,
    "anomaly_start": 39,
    "anomaly_end": 86,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 96,
    "subset": "val"
  },
  "sUbP4j8rAic_005473": {
    "video_start": 5473,
    "video_end": 5556,
    "anomaly_start": 35,
    "anomaly_end": 53,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 84,
    "subset": "val"
  },
  "sdqIDbZJfy0_001380": {
    "video_start": 1380,
    "video_end": 1489,
    "anomaly_start": 31,
    "anomaly_end": 44,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 110,
    "subset": "val"
  },
  "sdqIDbZJfy0_001846": {
    "video_start": 1846,
    "video_end": 1970,
    "anomaly_start": 44,
    "anomaly_end": 99,
    "anomaly_class": "ego: lateral",
    "num_frames": 125,
    "subset": "val"
  },
  "sdqIDbZJfy0_002337": {
    "video_start": 2337,
    "video_end": 2449,
    "anomaly_start": 35,
    "anomaly_end": 52,
    "anomaly_class": "other: turning",
    "num_frames": 113,
    "subset": "val"
  },
  "sdqIDbZJfy0_002811": {
    "video_start": 2811,
    "video_end": 2949,
    "anomaly_start": 61,
    "anomaly_end": 103,
    "anomaly_class": "ego: oncoming",
    "num_frames": 139,
    "subset": "val"
  },
  "t4Cvbdtebk0_000195": {
    "video_start": 195,
    "video_end": 343,
    "anomaly_start": 27,
    "anomaly_end": 134,
    "anomaly_class": "ego: lateral",
    "num_frames": 149,
    "subset": "val"
  },
  "t4Cvbdtebk0_000345": {
    "video_start": 345,
    "video_end": 443,
    "anomaly_start": 39,
    "anomaly_end": 80,
    "anomaly_class": "ego: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "t4Cvbdtebk0_001121": {
    "video_start": 1121,
    "video_end": 1260,
    "anomaly_start": 45,
    "anomaly_end": 77,
    "anomaly_class": "ego: turning",
    "num_frames": 140,
    "subset": "val"
  },
  "t4Cvbdtebk0_001262": {
    "video_start": 1262,
    "video_end": 1383,
    "anomaly_start": 37,
    "anomaly_end": 77,
    "anomaly_class": "other: turning",
    "num_frames": 122,
    "subset": "val"
  },
  "t4Cvbdtebk0_002644": {
    "video_start": 2644,
    "video_end": 2880,
    "anomaly_start": 103,
    "anomaly_end": 119,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 237,
    "subset": "val"
  },
  "t4Cvbdtebk0_003004": {
    "video_start": 3004,
    "video_end": 3094,
    "anomaly_start": 40,
    "anomaly_end": 63,
    "anomaly_class": "ego: obstacle",
    "num_frames": 91,
    "subset": "val"
  },
  "t4Cvbdtebk0_003225": {
    "video_start": 3225,
    "video_end": 3276,
    "anomaly_start": 44,
    "anomaly_end": 52,
    "anomaly_class": "ego: oncoming",
    "num_frames": 52,
    "subset": "val"
  },
  "t4Cvbdtebk0_003315": {
    "video_start": 3315,
    "video_end": 3403,
    "anomaly_start": 28,
    "anomaly_end": 47,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "t4Cvbdtebk0_003405": {
    "video_start": 3405,
    "video_end": 3503,
    "anomaly_start": 46,
    "anomaly_end": 80,
    "anomaly_class": "ego: oncoming",
    "num_frames": 99,
    "subset": "val"
  },
  "t4Cvbdtebk0_003795": {
    "video_start": 3795,
    "video_end": 3933,
    "anomaly_start": 31,
    "anomaly_end": 53,
    "anomaly_class": "other: turning",
    "num_frames": 139,
    "subset": "val"
  },
  "t4Cvbdtebk0_004215": {
    "video_start": 4215,
    "video_end": 4323,
    "anomaly_start": 43,
    "anomaly_end": 85,
    "anomaly_class": "ego: lateral",
    "num_frames": 109,
    "subset": "val"
  },
  "t4Cvbdtebk0_004325": {
    "video_start": 4325,
    "video_end": 4504,
    "anomaly_start": 76,
    "anomaly_end": 96,
    "anomaly_class": "ego: obstacle",
    "num_frames": 180,
    "subset": "val"
  },
  "t4Cvbdtebk0_004794": {
    "video_start": 4794,
    "video_end": 4946,
    "anomaly_start": 106,
    "anomaly_end": 114,
    "anomaly_class": "other: lateral",
    "num_frames": 153,
    "subset": "val"
  },
  "t4Cvbdtebk0_005158": {
    "video_start": 5158,
    "video_end": 5441,
    "anomaly_start": 140,
    "anomaly_end": 228,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 284,
    "subset": "val"
  },
  "t_bCDlNOhJc_000341": {
    "video_start": 341,
    "video_end": 429,
    "anomaly_start": 29,
    "anomaly_end": 54,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "t_bCDlNOhJc_000741": {
    "video_start": 741,
    "video_end": 840,
    "anomaly_start": 27,
    "anomaly_end": 58,
    "anomaly_class": "other: turning",
    "num_frames": 100,
    "subset": "val"
  },
  "t_bCDlNOhJc_002483": {
    "video_start": 2483,
    "video_end": 2591,
    "anomaly_start": 32,
    "anomaly_end": 74,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 109,
    "subset": "val"
  },
  "t_bCDlNOhJc_003675": {
    "video_start": 3675,
    "video_end": 3813,
    "anomaly_start": 62,
    "anomaly_end": 99,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 139,
    "subset": "val"
  },
  "t_bCDlNOhJc_003913": {
    "video_start": 3913,
    "video_end": 4061,
    "anomaly_start": 94,
    "anomaly_end": 109,
    "anomaly_class": "other: turning",
    "num_frames": 149,
    "subset": "val"
  },
  "t_bCDlNOhJc_004063": {
    "video_start": 4063,
    "video_end": 4163,
    "anomaly_start": 35,
    "anomaly_end": 74,
    "anomaly_class": "other: turning",
    "num_frames": 101,
    "subset": "val"
  },
  "t_bCDlNOhJc_004355": {
    "video_start": 4355,
    "video_end": 4473,
    "anomaly_start": 41,
    "anomaly_end": 57,
    "anomaly_class": "other: lateral",
    "num_frames": 119,
    "subset": "val"
  },
  "t_bCDlNOhJc_004475": {
    "video_start": 4475,
    "video_end": 4664,
    "anomaly_start": 59,
    "anomaly_end": 151,
    "anomaly_class": "ego: lateral",
    "num_frames": 190,
    "subset": "val"
  },
  "u33fdjUY_Iw_000076": {
    "video_start": 76,
    "video_end": 124,
    "anomaly_start": 38,
    "anomaly_end": 48,
    "anomaly_class": "ego: lateral",
    "num_frames": 49,
    "subset": "val"
  },
  "u33fdjUY_Iw_001302": {
    "video_start": 1302,
    "video_end": 1509,
    "anomaly_start": 103,
    "anomaly_end": 192,
    "anomaly_class": "other: lateral",
    "num_frames": 208,
    "subset": "val"
  },
  "u33fdjUY_Iw_001511": {
    "video_start": 1511,
    "video_end": 1591,
    "anomaly_start": 39,
    "anomaly_end": 67,
    "anomaly_class": "ego: turning",
    "num_frames": 81,
    "subset": "val"
  },
  "u33fdjUY_Iw_001746": {
    "video_start": 1746,
    "video_end": 1873,
    "anomaly_start": 31,
    "anomaly_end": 64,
    "anomaly_class": "ego: turning",
    "num_frames": 128,
    "subset": "val"
  },
  "u33fdjUY_Iw_002019": {
    "video_start": 2019,
    "video_end": 2130,
    "anomaly_start": 43,
    "anomaly_end": 107,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 112,
    "subset": "val"
  },
  "u33fdjUY_Iw_002325": {
    "video_start": 2325,
    "video_end": 2423,
    "anomaly_start": 46,
    "anomaly_end": 63,
    "anomaly_class": "other: oncoming",
    "num_frames": 99,
    "subset": "val"
  },
  "u33fdjUY_Iw_003228": {
    "video_start": 3228,
    "video_end": 3378,
    "anomaly_start": 38,
    "anomaly_end": 108,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 151,
    "subset": "val"
  },
  "u33fdjUY_Iw_005106": {
    "video_start": 5106,
    "video_end": 5265,
    "anomaly_start": 14,
    "anomaly_end": 50,
    "anomaly_class": "other: pedestrian",
    "num_frames": 160,
    "subset": "val"
  },
  "uFwnmh0GpBo_000291": {
    "video_start": 291,
    "video_end": 489,
    "anomaly_start": 150,
    "anomaly_end": 167,
    "anomaly_class": "other: turning",
    "num_frames": 199,
    "subset": "val"
  },
  "uFwnmh0GpBo_001232": {
    "video_start": 1232,
    "video_end": 1330,
    "anomaly_start": 57,
    "anomaly_end": 76,
    "anomaly_class": "other: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "uFwnmh0GpBo_002529": {
    "video_start": 2529,
    "video_end": 2667,
    "anomaly_start": 86,
    "anomaly_end": 105,
    "anomaly_class": "ego: lateral",
    "num_frames": 139,
    "subset": "val"
  },
  "uFwnmh0GpBo_003026": {
    "video_start": 3026,
    "video_end": 3114,
    "anomaly_start": 35,
    "anomaly_end": 58,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "uFwnmh0GpBo_004489": {
    "video_start": 4489,
    "video_end": 4607,
    "anomaly_start": 41,
    "anomaly_end": 60,
    "anomaly_class": "other: lateral",
    "num_frames": 119,
    "subset": "val"
  },
  "uFwnmh0GpBo_004979": {
    "video_start": 4979,
    "video_end": 5087,
    "anomaly_start": 38,
    "anomaly_end": 93,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 109,
    "subset": "val"
  },
  "uFwnmh0GpBo_005400": {
    "video_start": 5400,
    "video_end": 5488,
    "anomaly_start": 27,
    "anomaly_end": 52,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "uFwnmh0GpBo_005710": {
    "video_start": 5710,
    "video_end": 5892,
    "anomaly_start": 30,
    "anomaly_end": 140,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 183,
    "subset": "val"
  },
  "uO2zGO5ydBc_000547": {
    "video_start": 547,
    "video_end": 609,
    "anomaly_start": 10,
    "anomaly_end": 31,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 63,
    "subset": "val"
  },
  "uO2zGO5ydBc_000906": {
    "video_start": 906,
    "video_end": 984,
    "anomaly_start": 21,
    "anomaly_end": 39,
    "anomaly_class": "ego: turning",
    "num_frames": 79,
    "subset": "val"
  },
  "uO2zGO5ydBc_002272": {
    "video_start": 2272,
    "video_end": 2349,
    "anomaly_start": 32,
    "anomaly_end": 71,
    "anomaly_class": "ego: oncoming",
    "num_frames": 78,
    "subset": "val"
  },
  "uO2zGO5ydBc_002351": {
    "video_start": 2351,
    "video_end": 2531,
    "anomaly_start": 76,
    "anomaly_end": 138,
    "anomaly_class": "ego: lateral",
    "num_frames": 181,
    "subset": "val"
  },
  "uO2zGO5ydBc_002533": {
    "video_start": 2533,
    "video_end": 2588,
    "anomaly_start": 28,
    "anomaly_end": 45,
    "anomaly_class": "other: turning",
    "num_frames": 56,
    "subset": "val"
  },
  "uO2zGO5ydBc_002651": {
    "video_start": 2651,
    "video_end": 2718,
    "anomaly_start": 25,
    "anomaly_end": 53,
    "anomaly_class": "ego: obstacle",
    "num_frames": 68,
    "subset": "val"
  },
  "uO2zGO5ydBc_003373": {
    "video_start": 3373,
    "video_end": 3460,
    "anomaly_start": 17,
    "anomaly_end": 53,
    "anomaly_class": "ego: obstacle",
    "num_frames": 88,
    "subset": "val"
  },
  "uO2zGO5ydBc_003726": {
    "video_start": 3726,
    "video_end": 3792,
    "anomaly_start": 18,
    "anomaly_end": 41,
    "anomaly_class": "ego: turning",
    "num_frames": 67,
    "subset": "val"
  },
  "uO2zGO5ydBc_004625": {
    "video_start": 4625,
    "video_end": 4700,
    "anomaly_start": 14,
    "anomaly_end": 46,
    "anomaly_class": "ego: lateral",
    "num_frames": 76,
    "subset": "val"
  },
  "uO2zGO5ydBc_004956": {
    "video_start": 4956,
    "video_end": 5035,
    "anomaly_start": 46,
    "anomaly_end": 57,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 80,
    "subset": "val"
  },
  "ujVRDgGVzlQ_002028": {
    "video_start": 2028,
    "video_end": 2148,
    "anomaly_start": 40,
    "anomaly_end": 65,
    "anomaly_class": "ego: lateral",
    "num_frames": 121,
    "subset": "val"
  },
  "ujVRDgGVzlQ_002385": {
    "video_start": 2385,
    "video_end": 2468,
    "anomaly_start": 32,
    "anomaly_end": 45,
    "anomaly_class": "ego: turning",
    "num_frames": 84,
    "subset": "val"
  },
  "ujVRDgGVzlQ_002689": {
    "video_start": 2689,
    "video_end": 2775,
    "anomaly_start": 37,
    "anomaly_end": 65,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 87,
    "subset": "val"
  },
  "ujVRDgGVzlQ_003036": {
    "video_start": 3036,
    "video_end": 3140,
    "anomaly_start": 60,
    "anomaly_end": 74,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 105,
    "subset": "val"
  },
  "ujVRDgGVzlQ_003437": {
    "video_start": 3437,
    "video_end": 3545,
    "anomaly_start": 30,
    "anomaly_end": 62,
    "anomaly_class": "ego: oncoming",
    "num_frames": 109,
    "subset": "val"
  },
  "ujVRDgGVzlQ_003547": {
    "video_start": 3547,
    "video_end": 3625,
    "anomaly_start": 30,
    "anomaly_end": 59,
    "anomaly_class": "ego: oncoming",
    "num_frames": 79,
    "subset": "val"
  },
  "ujVRDgGVzlQ_003987": {
    "video_start": 3987,
    "video_end": 4097,
    "anomaly_start": 54,
    "anomaly_end": 79,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 111,
    "subset": "val"
  },
  "ujVRDgGVzlQ_005123": {
    "video_start": 5123,
    "video_end": 5194,
    "anomaly_start": 29,
    "anomaly_end": 58,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 72,
    "subset": "val"
  },
  "ujVRDgGVzlQ_005768": {
    "video_start": 5768,
    "video_end": 5854,
    "anomaly_start": 36,
    "anomaly_end": 57,
    "anomaly_class": "other: lateral",
    "num_frames": 87,
    "subset": "val"
  },
  "vZSr9dhRxlE_000175": {
    "video_start": 175,
    "video_end": 333,
    "anomaly_start": 43,
    "anomaly_end": 78,
    "anomaly_class": "other: turning",
    "num_frames": 159,
    "subset": "val"
  },
  "vZSr9dhRxlE_002810": {
    "video_start": 2810,
    "video_end": 2900,
    "anomaly_start": 31,
    "anomaly_end": 56,
    "anomaly_class": "ego: lateral",
    "num_frames": 91,
    "subset": "val"
  },
  "vZSr9dhRxlE_004531": {
    "video_start": 4531,
    "video_end": 4587,
    "anomaly_start": 18,
    "anomaly_end": 35,
    "anomaly_class": "ego: turning",
    "num_frames": 57,
    "subset": "val"
  },
  "vdLn-qswnRo_000604": {
    "video_start": 604,
    "video_end": 676,
    "anomaly_start": 1,
    "anomaly_end": 35,
    "anomaly_class": "other: lateral",
    "num_frames": 73,
    "subset": "val"
  },
  "vdLn-qswnRo_001232": {
    "video_start": 1232,
    "video_end": 1331,
    "anomaly_start": 43,
    "anomaly_end": 62,
    "anomaly_class": "ego: turning",
    "num_frames": 100,
    "subset": "val"
  },
  "vdLn-qswnRo_001333": {
    "video_start": 1333,
    "video_end": 1409,
    "anomaly_start": 26,
    "anomaly_end": 39,
    "anomaly_class": "other: obstacle",
    "num_frames": 77,
    "subset": "val"
  },
  "vdLn-qswnRo_001505": {
    "video_start": 1505,
    "video_end": 1609,
    "anomaly_start": 27,
    "anomaly_end": 48,
    "anomaly_class": "other: turning",
    "num_frames": 105,
    "subset": "val"
  },
  "vdLn-qswnRo_001707": {
    "video_start": 1707,
    "video_end": 1845,
    "anomaly_start": 36,
    "anomaly_end": 56,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 139,
    "subset": "val"
  },
  "vdLn-qswnRo_002457": {
    "video_start": 2457,
    "video_end": 2569,
    "anomaly_start": 31,
    "anomaly_end": 72,
    "anomaly_class": "ego: lateral",
    "num_frames": 113,
    "subset": "val"
  },
  "vdLn-qswnRo_004325": {
    "video_start": 4325,
    "video_end": 4415,
    "anomaly_start": 1,
    "anomaly_end": 39,
    "anomaly_class": "ego: turning",
    "num_frames": 91,
    "subset": "val"
  },
  "vdLn-qswnRo_004417": {
    "video_start": 4417,
    "video_end": 4488,
    "anomaly_start": 39,
    "anomaly_end": 62,
    "anomaly_class": "ego: lateral",
    "num_frames": 72,
    "subset": "val"
  },
  "vdLn-qswnRo_004935": {
    "video_start": 4935,
    "video_end": 5033,
    "anomaly_start": 51,
    "anomaly_end": 61,
    "anomaly_class": "ego: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "vdLn-qswnRo_005642": {
    "video_start": 5642,
    "video_end": 5724,
    "anomaly_start": 16,
    "anomaly_end": 66,
    "anomaly_class": "ego: turning",
    "num_frames": 83,
    "subset": "val"
  },
  "xToi8YVZM8o_000951": {
    "video_start": 951,
    "video_end": 1069,
    "anomaly_start": 37,
    "anomaly_end": 101,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 119,
    "subset": "val"
  },
  "xToi8YVZM8o_001071": {
    "video_start": 1071,
    "video_end": 1198,
    "anomaly_start": 49,
    "anomaly_end": 116,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 128,
    "subset": "val"
  },
  "xToi8YVZM8o_003151": {
    "video_start": 3151,
    "video_end": 3299,
    "anomaly_start": 43,
    "anomaly_end": 62,
    "anomaly_class": "other: pedestrian",
    "num_frames": 149,
    "subset": "val"
  },
  "xToi8YVZM8o_003642": {
    "video_start": 3642,
    "video_end": 3750,
    "anomaly_start": 27,
    "anomaly_end": 90,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 109,
    "subset": "val"
  },
  "xToi8YVZM8o_003752": {
    "video_start": 3752,
    "video_end": 3840,
    "anomaly_start": 35,
    "anomaly_end": 54,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 89,
    "subset": "val"
  },
  "xVfPBzDnI2Q_000992": {
    "video_start": 992,
    "video_end": 1105,
    "anomaly_start": 42,
    "anomaly_end": 88,
    "anomaly_class": "ego: lateral",
    "num_frames": 114,
    "subset": "val"
  },
  "xVfPBzDnI2Q_001107": {
    "video_start": 1107,
    "video_end": 1187,
    "anomaly_start": 37,
    "anomaly_end": 47,
    "anomaly_class": "ego: turning",
    "num_frames": 81,
    "subset": "val"
  },
  "xVfPBzDnI2Q_001454": {
    "video_start": 1454,
    "video_end": 1527,
    "anomaly_start": 17,
    "anomaly_end": 32,
    "anomaly_class": "ego: turning",
    "num_frames": 74,
    "subset": "val"
  },
  "xVfPBzDnI2Q_002332": {
    "video_start": 2332,
    "video_end": 2420,
    "anomaly_start": 39,
    "anomaly_end": 43,
    "anomaly_class": "ego: lateral",
    "num_frames": 89,
    "subset": "val"
  },
  "xVfPBzDnI2Q_002819": {
    "video_start": 2819,
    "video_end": 2896,
    "anomaly_start": 20,
    "anomaly_end": 31,
    "anomaly_class": "ego: pedestrian",
    "num_frames": 78,
    "subset": "val"
  },
  "xVfPBzDnI2Q_005805": {
    "video_start": 5805,
    "video_end": 5913,
    "anomaly_start": 14,
    "anomaly_end": 36,
    "anomaly_class": "ego: lateral",
    "num_frames": 109,
    "subset": "val"
  },
  "xjbbH74SwTg_000321": {
    "video_start": 321,
    "video_end": 409,
    "anomaly_start": 42,
    "anomaly_end": 68,
    "anomaly_class": "ego: lateral",
    "num_frames": 89,
    "subset": "val"
  },
  "xjbbH74SwTg_000512": {
    "video_start": 512,
    "video_end": 630,
    "anomaly_start": 69,
    "anomaly_end": 89,
    "anomaly_class": "other: turning",
    "num_frames": 119,
    "subset": "val"
  },
  "xjbbH74SwTg_001419": {
    "video_start": 1419,
    "video_end": 1507,
    "anomaly_start": 30,
    "anomaly_end": 51,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "xjbbH74SwTg_002050": {
    "video_start": 2050,
    "video_end": 2138,
    "anomaly_start": 41,
    "anomaly_end": 53,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "xjbbH74SwTg_002647": {
    "video_start": 2647,
    "video_end": 2724,
    "anomaly_start": 67,
    "anomaly_end": 78,
    "anomaly_class": "ego: turning",
    "num_frames": 78,
    "subset": "val"
  },
  "xjbbH74SwTg_002977": {
    "video_start": 2977,
    "video_end": 3055,
    "anomaly_start": 41,
    "anomaly_end": 55,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 79,
    "subset": "val"
  },
  "xjbbH74SwTg_004466": {
    "video_start": 4466,
    "video_end": 4534,
    "anomaly_start": 31,
    "anomaly_end": 47,
    "anomaly_class": "other: oncoming",
    "num_frames": 69,
    "subset": "val"
  },
  "xjbbH74SwTg_004536": {
    "video_start": 4536,
    "video_end": 4624,
    "anomaly_start": 34,
    "anomaly_end": 58,
    "anomaly_class": "other: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "xjbbH74SwTg_004766": {
    "video_start": 4766,
    "video_end": 4865,
    "anomaly_start": 48,
    "anomaly_end": 72,
    "anomaly_class": "other: lateral",
    "num_frames": 100,
    "subset": "val"
  },
  "xpOyD-qrQUw_000236": {
    "video_start": 236,
    "video_end": 334,
    "anomaly_start": 49,
    "anomaly_end": 82,
    "anomaly_class": "ego: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "xpOyD-qrQUw_000470": {
    "video_start": 470,
    "video_end": 580,
    "anomaly_start": 31,
    "anomaly_end": 45,
    "anomaly_class": "ego: lateral",
    "num_frames": 111,
    "subset": "val"
  },
  "xpOyD-qrQUw_000890": {
    "video_start": 890,
    "video_end": 982,
    "anomaly_start": 40,
    "anomaly_end": 69,
    "anomaly_class": "ego: turning",
    "num_frames": 93,
    "subset": "val"
  },
  "xpOyD-qrQUw_002724": {
    "video_start": 2724,
    "video_end": 2805,
    "anomaly_start": 11,
    "anomaly_end": 27,
    "anomaly_class": "other: lateral",
    "num_frames": 82,
    "subset": "val"
  },
  "xpOyD-qrQUw_002906": {
    "video_start": 2906,
    "video_end": 3001,
    "anomaly_start": 38,
    "anomaly_end": 58,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 96,
    "subset": "val"
  },
  "xpOyD-qrQUw_003374": {
    "video_start": 3374,
    "video_end": 3461,
    "anomaly_start": 51,
    "anomaly_end": 67,
    "anomaly_class": "other: turning",
    "num_frames": 88,
    "subset": "val"
  },
  "xpOyD-qrQUw_004619": {
    "video_start": 4619,
    "video_end": 4672,
    "anomaly_start": 40,
    "anomaly_end": 54,
    "anomaly_class": "ego: turning",
    "num_frames": 54,
    "subset": "val"
  },
  "xpOyD-qrQUw_004837": {
    "video_start": 4837,
    "video_end": 4894,
    "anomaly_start": 20,
    "anomaly_end": 34,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 58,
    "subset": "val"
  },
  "xpOyD-qrQUw_005108": {
    "video_start": 5108,
    "video_end": 5172,
    "anomaly_start": 14,
    "anomaly_end": 40,
    "anomaly_class": "other: turning",
    "num_frames": 65,
    "subset": "val"
  },
  "y1vGuUK0db4_000230": {
    "video_start": 230,
    "video_end": 318,
    "anomaly_start": 26,
    "anomaly_end": 59,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 89,
    "subset": "val"
  },
  "y1vGuUK0db4_001096": {
    "video_start": 1096,
    "video_end": 1186,
    "anomaly_start": 19,
    "anomaly_end": 77,
    "anomaly_class": "ego: turning",
    "num_frames": 91,
    "subset": "val"
  },
  "y1vGuUK0db4_001404": {
    "video_start": 1404,
    "video_end": 1542,
    "anomaly_start": 47,
    "anomaly_end": 77,
    "anomaly_class": "other: lateral",
    "num_frames": 139,
    "subset": "val"
  },
  "y1vGuUK0db4_002066": {
    "video_start": 2066,
    "video_end": 2169,
    "anomaly_start": 19,
    "anomaly_end": 34,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 104,
    "subset": "val"
  },
  "y1vGuUK0db4_002278": {
    "video_start": 2278,
    "video_end": 2358,
    "anomaly_start": 24,
    "anomaly_end": 49,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 81,
    "subset": "val"
  },
  "y1vGuUK0db4_002468": {
    "video_start": 2468,
    "video_end": 2567,
    "anomaly_start": 44,
    "anomaly_end": 73,
    "anomaly_class": "ego: turning",
    "num_frames": 100,
    "subset": "val"
  },
  "y1vGuUK0db4_002709": {
    "video_start": 2709,
    "video_end": 2812,
    "anomaly_start": 43,
    "anomaly_end": 66,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 104,
    "subset": "val"
  },
  "y1vGuUK0db4_003670": {
    "video_start": 3670,
    "video_end": 3777,
    "anomaly_start": 26,
    "anomaly_end": 65,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 108,
    "subset": "val"
  },
  "y1vGuUK0db4_004216": {
    "video_start": 4216,
    "video_end": 4303,
    "anomaly_start": 27,
    "anomaly_end": 45,
    "anomaly_class": "ego: lateral",
    "num_frames": 88,
    "subset": "val"
  },
  "y1vGuUK0db4_004305": {
    "video_start": 4305,
    "video_end": 4388,
    "anomaly_start": 32,
    "anomaly_end": 50,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 84,
    "subset": "val"
  },
  "y1vGuUK0db4_004495": {
    "video_start": 4495,
    "video_end": 4572,
    "anomaly_start": 22,
    "anomaly_end": 41,
    "anomaly_class": "ego: turning",
    "num_frames": 78,
    "subset": "val"
  },
  "y1vGuUK0db4_005716": {
    "video_start": 5716,
    "video_end": 5888,
    "anomaly_start": 20,
    "anomaly_end": 53,
    "anomaly_class": "other: lateral",
    "num_frames": 173,
    "subset": "val"
  },
  "y4Evv5By6sg_001536": {
    "video_start": 1536,
    "video_end": 1576,
    "anomaly_start": 34,
    "anomaly_end": 41,
    "anomaly_class": "ego: oncoming",
    "num_frames": 41,
    "subset": "val"
  },
  "y4Evv5By6sg_004171": {
    "video_start": 4171,
    "video_end": 4264,
    "anomaly_start": 8,
    "anomaly_end": 82,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 94,
    "subset": "val"
  },
  "y4Evv5By6sg_004266": {
    "video_start": 4266,
    "video_end": 4364,
    "anomaly_start": 34,
    "anomaly_end": 78,
    "anomaly_class": "other: oncoming",
    "num_frames": 99,
    "subset": "val"
  },
  "y4Evv5By6sg_004366": {
    "video_start": 4366,
    "video_end": 4464,
    "anomaly_start": 32,
    "anomaly_end": 84,
    "anomaly_class": "ego: lateral",
    "num_frames": 99,
    "subset": "val"
  },
  "y4Evv5By6sg_004695": {
    "video_start": 4695,
    "video_end": 4803,
    "anomaly_start": 31,
    "anomaly_end": 68,
    "anomaly_class": "other: turning",
    "num_frames": 109,
    "subset": "val"
  },
  "y4Evv5By6sg_005070": {
    "video_start": 5070,
    "video_end": 5188,
    "anomaly_start": 52,
    "anomaly_end": 101,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 119,
    "subset": "val"
  },
  "y4Evv5By6sg_005190": {
    "video_start": 5190,
    "video_end": 5317,
    "anomaly_start": 38,
    "anomaly_end": 111,
    "anomaly_class": "ego: leave_to_left",
    "num_frames": 128,
    "subset": "val"
  },
  "yB6fjM1UUC0_000241": {
    "video_start": 241,
    "video_end": 339,
    "anomaly_start": 33,
    "anomaly_end": 61,
    "anomaly_class": "other: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "yB6fjM1UUC0_000957": {
    "video_start": 957,
    "video_end": 1055,
    "anomaly_start": 31,
    "anomaly_end": 57,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 99,
    "subset": "val"
  },
  "yB6fjM1UUC0_001177": {
    "video_start": 1177,
    "video_end": 1295,
    "anomaly_start": 29,
    "anomaly_end": 49,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 119,
    "subset": "val"
  },
  "yB6fjM1UUC0_001713": {
    "video_start": 1713,
    "video_end": 1807,
    "anomaly_start": 21,
    "anomaly_end": 65,
    "anomaly_class": "other: turning",
    "num_frames": 95,
    "subset": "val"
  },
  "yhtzAKqRyXw_000209": {
    "video_start": 209,
    "video_end": 326,
    "anomaly_start": 19,
    "anomaly_end": 37,
    "anomaly_class": "other: turning",
    "num_frames": 118,
    "subset": "val"
  },
  "yhtzAKqRyXw_000506": {
    "video_start": 506,
    "video_end": 607,
    "anomaly_start": 22,
    "anomaly_end": 63,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 102,
    "subset": "val"
  },
  "yhtzAKqRyXw_001192": {
    "video_start": 1192,
    "video_end": 1296,
    "anomaly_start": 11,
    "anomaly_end": 67,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 105,
    "subset": "val"
  },
  "yhtzAKqRyXw_001424": {
    "video_start": 1424,
    "video_end": 1492,
    "anomaly_start": 25,
    "anomaly_end": 40,
    "anomaly_class": "other: turning",
    "num_frames": 69,
    "subset": "val"
  },
  "yhtzAKqRyXw_001494": {
    "video_start": 1494,
    "video_end": 1648,
    "anomaly_start": 57,
    "anomaly_end": 86,
    "anomaly_class": "ego: oncoming",
    "num_frames": 155,
    "subset": "val"
  },
  "yhtzAKqRyXw_002226": {
    "video_start": 2226,
    "video_end": 2309,
    "anomaly_start": 28,
    "anomaly_end": 42,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 84,
    "subset": "val"
  },
  "yhtzAKqRyXw_002878": {
    "video_start": 2878,
    "video_end": 2992,
    "anomaly_start": 10,
    "anomaly_end": 57,
    "anomaly_class": "ego: lateral",
    "num_frames": 115,
    "subset": "val"
  },
  "yhtzAKqRyXw_003574": {
    "video_start": 3574,
    "video_end": 3657,
    "anomaly_start": 25,
    "anomaly_end": 42,
    "anomaly_class": "ego: turning",
    "num_frames": 84,
    "subset": "val"
  },
  "yhtzAKqRyXw_004036": {
    "video_start": 4036,
    "video_end": 4169,
    "anomaly_start": 15,
    "anomaly_end": 72,
    "anomaly_class": "other: leave_to_left",
    "num_frames": 134,
    "subset": "val"
  },
  "yhtzAKqRyXw_004362": {
    "video_start": 4362,
    "video_end": 4478,
    "anomaly_start": 43,
    "anomaly_end": 68,
    "anomaly_class": "other: lateral",
    "num_frames": 117,
    "subset": "val"
  },
  "yhtzAKqRyXw_004558": {
    "video_start": 4558,
    "video_end": 4641,
    "anomaly_start": 30,
    "anomaly_end": 47,
    "anomaly_class": "ego: turning",
    "num_frames": 84,
    "subset": "val"
  },
  "yhtzAKqRyXw_004643": {
    "video_start": 4643,
    "video_end": 4731,
    "anomaly_start": 28,
    "anomaly_end": 52,
    "anomaly_class": "ego: turning",
    "num_frames": 89,
    "subset": "val"
  },
  "yhtzAKqRyXw_004733": {
    "video_start": 4733,
    "video_end": 4809,
    "anomaly_start": 16,
    "anomaly_end": 56,
    "anomaly_class": "other: turning",
    "num_frames": 77,
    "subset": "val"
  },
  "yhtzAKqRyXw_004811": {
    "video_start": 4811,
    "video_end": 4934,
    "anomaly_start": 52,
    "anomaly_end": 86,
    "anomaly_class": "other: start_stop_or_stationary",
    "num_frames": 124,
    "subset": "val"
  },
  "yhtzAKqRyXw_004936": {
    "video_start": 4936,
    "video_end": 5036,
    "anomaly_start": 22,
    "anomaly_end": 49,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 101,
    "subset": "val"
  },
  "yhtzAKqRyXw_005038": {
    "video_start": 5038,
    "video_end": 5112,
    "anomaly_start": 34,
    "anomaly_end": 42,
    "anomaly_class": "ego: oncoming",
    "num_frames": 75,
    "subset": "val"
  },
  "yr0gNIxRluE_000613": {
    "video_start": 613,
    "video_end": 721,
    "anomaly_start": 52,
    "anomaly_end": 88,
    "anomaly_class": "ego: leave_to_right",
    "num_frames": 109,
    "subset": "val"
  },
  "yr0gNIxRluE_000723": {
    "video_start": 723,
    "video_end": 872,
    "anomaly_start": 24,
    "anomaly_end": 46,
    "anomaly_class": "other: lateral",
    "num_frames": 150,
    "subset": "val"
  },
  "yr0gNIxRluE_001170": {
    "video_start": 1170,
    "video_end": 1273,
    "anomaly_start": 64,
    "anomaly_end": 74,
    "anomaly_class": "other: turning",
    "num_frames": 104,
    "subset": "val"
  },
  "yr0gNIxRluE_001361": {
    "video_start": 1361,
    "video_end": 1494,
    "anomaly_start": 43,
    "anomaly_end": 94,
    "anomaly_class": "other: turning",
    "num_frames": 134,
    "subset": "val"
  },
  "yr0gNIxRluE_003972": {
    "video_start": 3972,
    "video_end": 4061,
    "anomaly_start": 20,
    "anomaly_end": 38,
    "anomaly_class": "other: oncoming",
    "num_frames": 90,
    "subset": "val"
  },
  "zRZ9PJguIfE_000193": {
    "video_start": 193,
    "video_end": 351,
    "anomaly_start": 65,
    "anomaly_end": 159,
    "anomaly_class": "other: lateral",
    "num_frames": 159,
    "subset": "val"
  },
  "zRZ9PJguIfE_000455": {
    "video_start": 455,
    "video_end": 553,
    "anomaly_start": 65,
    "anomaly_end": 96,
    "anomaly_class": "ego: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "zRZ9PJguIfE_000735": {
    "video_start": 735,
    "video_end": 834,
    "anomaly_start": 36,
    "anomaly_end": 51,
    "anomaly_class": "other: moving_ahead_or_waiting",
    "num_frames": 100,
    "subset": "val"
  },
  "zRZ9PJguIfE_000836": {
    "video_start": 836,
    "video_end": 934,
    "anomaly_start": 31,
    "anomaly_end": 49,
    "anomaly_class": "ego: moving_ahead_or_waiting",
    "num_frames": 99,
    "subset": "val"
  },
  "zRZ9PJguIfE_002271": {
    "video_start": 2271,
    "video_end": 2409,
    "anomaly_start": 57,
    "anomaly_end": 66,
    "anomaly_class": "other: oncoming",
    "num_frames": 139,
    "subset": "val"
  },
  "zRZ9PJguIfE_003237": {
    "video_start": 3237,
    "video_end": 3462,
    "anomaly_start": 48,
    "anomaly_end": 66,
    "anomaly_class": "other: start_stop_or_stationary",
    "num_frames": 226,
    "subset": "val"
  },
  "zRZ9PJguIfE_003464": {
    "video_start": 3464,
    "video_end": 3562,
    "anomaly_start": 39,
    "anomaly_end": 75,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 99,
    "subset": "val"
  },
  "zRZ9PJguIfE_003666": {
    "video_start": 3666,
    "video_end": 3764,
    "anomaly_start": 42,
    "anomaly_end": 58,
    "anomaly_class": "other: turning",
    "num_frames": 99,
    "subset": "val"
  },
  "zRZ9PJguIfE_004415": {
    "video_start": 4415,
    "video_end": 4503,
    "anomaly_start": 17,
    "anomaly_end": 36,
    "anomaly_class": "other: leave_to_right",
    "num_frames": 89,
    "subset": "val"
  },
  "zRZ9PJguIfE_004505": {
    "video_start": 4505,
    "video_end": 4613,
    "anomaly_start": 41,
    "anomaly_end": 94,
    "anomaly_class": "other: lateral",
    "num_frames": 109,
    "subset": "val"
  }
}
'''

# Parse JSON
parsed_data = json.loads(data)

# Prepare a list for storing rows
rows = []

# Loop through each video entry
for video_id, video_info in parsed_data.items():
    rows.append({
        "Video ID": video_id,
        "Video Start": video_info['video_start'],
        "Video End": video_info['video_end'],
        "Anomaly Start": video_info['anomaly_start'],
        "Anomaly End": video_info['anomaly_end'],
        "Anomaly Class": video_info['anomaly_class'],
        "Number of Frames": video_info['num_frames'],
        "Subset": video_info['subset']
    })

# Convert to DataFrame
df = pd.DataFrame(rows)

# Save to Excel
df.to_excel("traffic_anomaly_data_val.xlsx", index=False)

print("Conversion complete! Excel file saved.")
