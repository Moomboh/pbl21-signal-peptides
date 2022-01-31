from math import sqrt, isnan
from decimal import Decimal

import sys
sys.path.append('../consensus_model')
from consensus_model.model_interface import Model

from sklearn import metrics

_positives_dict = {
    'I': False,
    'M': False,
    'O': False,
    'S': True,
    'L': True,
    'T': True,
    'X': False
}

_character_index_dict = {
    'S': 0,
    'L': 1,
    'T': 2,
    'X': 3
}

_rnd_baseline_4_state_dict = {
    'S_acc': 0.8995115162441335,
    'S_pre': 0.053060049212083134,
    'S_rec': 0.053060049212083134,
    'S_mcc': 0.0,
    'L_acc': 0.9577615027021471,
    'L_pre': 0.02158805739656393,
    'L_rec': 0.02158805739656393,
    'L_mcc': 0.0,
    'T_acc': 0.9773980774546066,
    'T_pre': 0.011420934329627605,
    'T_rec': 0.011420934329627605,
    'T_mcc': 0.0,
    'X_acc': 0.8426629517641389,
    'X_pre': 0.9139222154087441,
    'X_rec': 0.9139222154087441,
    'X_mcc': 0.0,
    'all_acc': 0.9193335120412566,
    'all_pre': 0.2499978140867547,
    'all_rec': 0.2499978140867547,
    'all_mcc': 0.0,
}


# target	        group	label	accuracy	        precision	         recall	                accuracy_sd	            precision_sd	        recall_sd	            accuracy_ci	            precision_ci	        recall_ci
# residue_4state	_all_	L	    0.9577615027021471	0.02158805739656393	 0.02158805739656393	0.00012308729051554282	0.0008050449016740209	0.0007969068376507391	0.0002412461659188433	0.001577855805485014	0.0015619055255219425
# residue_4state	_all_	S	    0.8995115162441335	0.053060049212083134 0.053060049212083134	0.00018026500211462432	0.0008248599884360624	0.0007919809470907247	0.00035331219354457907	0.001616692582935145	0.0015522509770599367
# residue_4state	_all_	T	    0.9773980774546066	0.011420934329627605 0.011420934329627605	8.709633864736893e-05	0.0008025452034954294	0.0008007771043085875	0.0001707053398952972	0.0015729564970429018	0.0015694910933606592
# residue_4state	_all_	X	    0.8426629517641389	0.9139222154087441	 0.9139222154087441	    0.00023010496008978764	0.00024131992852091091	7.080342719131014e-05	0.00045099651757758014	0.0004729774071038445	0.00013877188515788022
# residue_4state	_all_	_all_	0.9193335120412566	0.2499978140867547	 0.2499978140867547	    0.00011191164705161925	0.00036382714133939277	0.0003637616672772222	0.00021934235175529167	0.0007130866439395562	0.0007129583173966644

# Tool in order to analyse given models
class Analysis:

    def __init__(self, model: Model):
        self.__model: Model = model
        self.__data_train_observation, self.__data_train_states = model.get_train_data()
        self.__data_test_observation, self.__data_test_states = model.get_test_data()
        self.__data_prediction_scored: [([str], float)] = [self.__model.annotate(observation) for observation in self.__data_test_observation]
        self.__data_predicted: [[str]] = [prediction for prediction, score in self.__data_prediction_scored]

    def accuracy(self) -> float:
        return metrics.accuracy_score(y_true=sum(self.__data_test_states, []), y_pred=sum(self.__data_predicted, []))

    def accuracy_per_state(self, state: str) -> float:
        return metrics.accuracy_score(y_true=['Q' if state == character else 'P' for character in sum(self.__data_test_states, [])],
                                      y_pred=['Q' if state == character else 'P' for character in sum(self.__data_predicted, [])])

    def precision(self) -> float:
        return metrics.precision_score(y_true=sum(self.__data_test_states, []), y_pred=sum(self.__data_predicted, []), average='macro')

    def precision_per_state(self, state: str) -> float:
        return metrics.precision_score(y_true=['Q' if state == character else 'P' for character in sum(self.__data_test_states, [])],
                                       y_pred=['Q' if state == character else 'P' for character in sum(self.__data_predicted, [])], average='macro')

    def mcc(self) -> float:
        return metrics.matthews_corrcoef(y_true=sum(self.__data_test_states, []), y_pred=sum(self.__data_predicted, []))

    def mcc_per_state(self, state: str) -> float:
        return metrics.matthews_corrcoef(y_true=['Q' if state == character else 'P' for character in sum(self.__data_test_states, [])],
                                         y_pred=['Q' if state == character else 'P' for character in sum(self.__data_predicted, [])])

    def recall(self) -> float:
        return metrics.recall_score(y_true=sum(self.__data_test_states, []), y_pred=sum(self.__data_predicted, []), average='macro')

    def recall_per_state(self, state: str) -> float:
        return metrics.recall_score(y_true=['Q' if state == character else 'P' for character in sum(self.__data_test_states, [])],
                                    y_pred=['Q' if state == character else 'P' for character in sum(self.__data_predicted, [])], average='macro')

    def tp_tn_fp_fn(self, _class: str = None) -> (int, int, int, int):
        true_positives: int = 0
        true_negatives: int = 0
        false_positives: int = 0
        false_negatives: int = 0
        for (predicted_state, true_state) in zip(self.__data_predicted, self.__data_test_states):
            for (predicted_character, true_character) in zip(predicted_state, true_state):
                if predicted_character == true_character:
                    if (_class is None and _positives_dict[predicted_character]) or true_character == _class:
                        true_positives += 1
                    else:
                        true_negatives += 1
                else:
                    if (_class is None and _positives_dict[predicted_character]) or true_character == _class:
                        false_positives += 1
                    else:
                        false_negatives += 1
        return true_positives, true_negatives, false_positives, false_negatives

    # def tp_tn_fp_fn_sklearn(self, _class: str = None) -> (int, int, int, int):
    #     cm = metrics.confusion_matrix()
    #     print(cm)
        # return true_positives, true_negatives, false_positives, false_negatives

    @staticmethod
    def accuracy_for_tp_tn_fp_fn(tp_tn_fp_fn: (int, int, int, int)) -> float:
        tp, tn, fp, fn = tp_tn_fp_fn
        numerator = tp + tn
        denominator = tp + tn + fp + fn
        return float('nan') if numerator == 0 or denominator == 0 else float(numerator) / float(denominator)

    @staticmethod
    def precision_for_tp_tn_fp_fn(tp_tn_fp_fn: (int, int, int, int)) -> float:
        tp, tn, fp, fn = tp_tn_fp_fn
        numerator = tp
        denominator = tp + fp
        return float('nan') if numerator == 0 or denominator == 0 else float(numerator) / float(denominator)

    @staticmethod
    def recall_for_tp_tn_fp_fn(tp_tn_fp_fn: (int, int, int, int)) -> float:
        tp, tn, fp, fn = tp_tn_fp_fn
        numerator = tp
        denominator = tp + fn
        return float('nan') if numerator == 0 or denominator == 0 else float(numerator) / float(denominator)

    @staticmethod
    def mcc_for_tp_tn_fp_fn(tp_tn_fp_fn: (int, int, int, int)) -> float:
        tp, tn, fp, fn = tp_tn_fp_fn
        tp = Decimal(str(tp))
        tn = Decimal(str(tp))
        fp = Decimal(str(tp))
        fn = Decimal(str(tp))
        numerator: Decimal = (tp * tn) - (fp * fn)
        denominator: Decimal = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        return 0.0 if numerator == 0 or denominator == 0 else float(numerator / (denominator.sqrt()))

    def confusion_matrix_types(self):
        matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        for (predicted_state, true_state) in zip(self.__data_predicted, self.__data_test_states):
            for (predicted_character, true_character) in zip(predicted_state, true_state):
                matrix[_character_index_dict[predicted_character]][_character_index_dict[true_character]] += 1
        return matrix

    @staticmethod
    def _rnd_baseline_check(key: str, value: float) -> bool:
        return _rnd_baseline_4_state_dict[key] < value if not isnan(value) else None


    @staticmethod
    def get_header():
        return "ACC_all,ACC_CHECK_all," \
               "PRE_all,PRE_CHECK_all," \
               "MCC_all,MCC_CHECK_all," \
               "REC_all,REC_CHECK_all," \
               "TP_all,TN_all,FP_all,FN_all," \
                \
               "ACC_S,ACC_CHECK_S," \
               "PRE_S,PRE_CHECK_S," \
               "MCC_S,MCC_CHECK_S," \
               "REC_S,REC_CHECK_S," \
               "TP_S,TN_S,FP_S,FN_S," \
                \
               "ACC_L,ACC_CHECK_L," \
               "PRE_L,PRE_CHECK_L," \
               "MCC_L,MCC_CHECK_L," \
               "REC_L,REC_CHECK_L," \
               "TP_L,TN_L,FP_L,FN_L," \
                \
               "ACC_T,ACC_CHECK_T," \
               "PRE_T,PRE_CHECK_T," \
               "MCC_T,MCC_CHECK_T," \
               "REC_T,REC_CHECK_T," \
               "TP_T,TN_T,FP_T,FN_T," \
                \
               "ACC_X,ACC_CHECK_X," \
               "PRE_X,PRE_CHECK_X," \
               "MCC_X,MCC_CHECK_X," \
               "REC_X,REC_CHECK_X," \
               "TP_X,TN_X,FP_X,FN_X"

    def __str__(self):
        tp_tn_fp_fn_all = self.tp_tn_fp_fn()
        tp_tn_fp_fn_s = self.tp_tn_fp_fn("S")
        tp_tn_fp_fn_l = self.tp_tn_fp_fn("L")
        tp_tn_fp_fn_t = self.tp_tn_fp_fn("T")
        tp_tn_fp_fn_x = self.tp_tn_fp_fn("X")

        all_acc = self.accuracy()
        all_pre = self.precision()
        all_mcc = self.mcc()
        all_rec = self.recall()

        S_acc = self.accuracy_per_state('S')
        S_pre = self.precision_per_state('S')
        S_mcc = self.mcc_per_state('S')
        S_rec = self.recall_per_state('S')

        L_acc = self.accuracy_per_state('L')
        L_pre = self.precision_per_state('L')
        L_mcc = self.mcc_per_state('L')
        L_rec = self.recall_per_state('L')

        T_acc = self.accuracy_per_state('T')
        T_pre = self.precision_per_state('T')
        T_mcc = self.mcc_per_state('T')
        T_rec = self.recall_per_state('T')

        X_acc = self.accuracy_per_state('X')
        X_pre = self.precision_per_state('X')
        X_mcc = self.mcc_per_state('X')
        X_rec = self.recall_per_state('X')

        return ','.join([
            str(all_acc), str(Analysis._rnd_baseline_check('all_acc', all_acc)),
            str(all_pre), str(Analysis._rnd_baseline_check('all_pre', all_pre)),
            str(all_mcc), str(Analysis._rnd_baseline_check('all_mcc', all_mcc)),
            str(all_rec), str(Analysis._rnd_baseline_check('all_rec', all_rec)),
            ','.join([str(x) for x in tp_tn_fp_fn_all]),

            str(S_acc), str(Analysis._rnd_baseline_check('S_acc', S_acc)),
            str(S_pre), str(Analysis._rnd_baseline_check('S_pre', S_pre)),
            str(S_mcc), str(Analysis._rnd_baseline_check('S_mcc', S_mcc)),
            str(S_rec), str(Analysis._rnd_baseline_check('S_rec', S_rec)),
            ','.join([str(x) for x in tp_tn_fp_fn_s]),

            str(L_acc), str(Analysis._rnd_baseline_check('L_acc', L_acc)),
            str(L_pre), str(Analysis._rnd_baseline_check('L_pre', L_pre)),
            str(L_mcc), str(Analysis._rnd_baseline_check('L_mcc', L_mcc)),
            str(L_rec), str(Analysis._rnd_baseline_check('L_rec', L_rec)),
            ','.join([str(x) for x in tp_tn_fp_fn_l]),

            str(T_acc), str(Analysis._rnd_baseline_check('T_acc', T_acc)),
            str(T_pre), str(Analysis._rnd_baseline_check('T_pre', T_pre)),
            str(T_mcc), str(Analysis._rnd_baseline_check('T_mcc', T_mcc)),
            str(T_rec), str(Analysis._rnd_baseline_check('T_rec', T_rec)),
            ','.join([str(x) for x in tp_tn_fp_fn_t]),

            str(X_acc), str(Analysis._rnd_baseline_check('X_acc', X_acc)),
            str(X_pre), str(Analysis._rnd_baseline_check('X_pre', X_pre)),
            str(X_mcc), str(Analysis._rnd_baseline_check('X_mcc', X_mcc)),
            str(X_rec), str(Analysis._rnd_baseline_check('X_rec', X_rec)),
            ','.join([str(x) for x in tp_tn_fp_fn_x]),
        ])
