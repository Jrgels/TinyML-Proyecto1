#pragma once
#include <stdint.h>

namespace har {

constexpr int NUM_FEATURES = 30;

inline int predict(const float features[NUM_FEATURES]) {
    // mean_total_acc_x
    if (features[0] <= 0.3867424875497818) {
        return 6; // LAYING
    } else {
        // std_total_acc_x
        if (features[6] <= 0.093192871659994125) {
            // mean_total_acc_y
            if (features[1] <= -0.066491559147834778) {
                // min_body_gyro_x
                if (features[15] <= -0.0094097941182553768) {
                    // min_total_acc_z
                    if (features[14] <= -0.081149674952030182) {
                        return 5; // STANDING
                    } else {
                        return 5; // STANDING
                    }
                } else {
                    // rms_total_acc_z
                    if (features[26] <= 0.10329263657331467) {
                        return 5; // STANDING
                    } else {
                        return 4; // SITTING
                    }
                }
            } else {
                // mean_total_acc_x
                if (features[0] <= 1.003227710723877) {
                    // std_body_gyro_z
                    if (features[11] <= 0.092413898557424545) {
                        return 4; // SITTING
                    } else {
                        return 4; // SITTING
                    }
                } else {
                    // max_body_gyro_x
                    if (features[21] <= 0.044252095744013786) {
                        return 4; // SITTING
                    } else {
                        return 5; // STANDING
                    }
                }
            }
        } else {
            // max_total_acc_x
            if (features[18] <= 1.8300740122795105) {
                // mean_total_acc_y
                if (features[1] <= -0.27936278283596039) {
                    // min_body_gyro_x
                    if (features[15] <= -1.1065235137939453) {
                        return 2; // WALKING_UPSTAIRS
                    } else {
                        return 2; // WALKING_UPSTAIRS
                    }
                } else {
                    // mean_body_gyro_x
                    if (features[3] <= 0.1107134148478508) {
                        return 1; // WALKING
                    } else {
                        return 2; // WALKING_UPSTAIRS
                    }
                }
            } else {
                // std_total_acc_x
                if (features[6] <= 0.33430276811122894) {
                    // mean_total_acc_y
                    if (features[1] <= -0.29112689197063446) {
                        return 2; // WALKING_UPSTAIRS
                    } else {
                        return 3; // WALKING_DOWNSTAIRS
                    }
                } else {
                    // mean_total_acc_y
                    if (features[1] <= -0.27038663625717163) {
                        return 3; // WALKING_DOWNSTAIRS
                    } else {
                        return 3; // WALKING_DOWNSTAIRS
                    }
                }
            }
        }
    }
}

inline const char* activityName(int clase) {
    switch (clase) {
        case 1: return "WALKING";
        case 2: return "WALKING_UPSTAIRS";
        case 3: return "WALKING_DOWNSTAIRS";
        case 4: return "SITTING";
        case 5: return "STANDING";
        case 6: return "LAYING";
        default: return "UNKNOWN";
    }
}

} // namespace har
