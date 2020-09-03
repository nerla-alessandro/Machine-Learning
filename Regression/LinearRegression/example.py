from linear_regression import LinearRegressionModel
import pandas as pd

crimeDataset = pd.read_csv("../Datasets/crimeDataset.data")
print("Crime Model:")
crimeModel = LinearRegressionModel(dataset=crimeDataset,
                                   predictor_variables=["RacePctBlack", "RacePctWhite", "RacePctAsian",
                                                        "RacePctHisp", "MedIncome", "PctPopUnderPov",
                                                        "PctLess9thGrade", "PctNotHSGrad", "PctUnemployed",
                                                        "MedRentPctHousInc", "MedOwnCostPctInc",
                                                        "LemasSwFTFieldPerPop", "LemasTotReqPerPop",
                                                        "LemasPctPolicOnPatr", "LemasGangUnitDeploy",
                                                        "LemasPctOfficDrugUn", "PolicBudgPerPop"],
                                   response_variables=["ViolentCrimesPerPop"],
                                   test_size_percentage=0.1,
                                   nan_val="?",
                                   target_accuracy=.9)
crimeModel.print_dataset_data()

autoCrimeDataset = pd.read_csv("../Datasets/crimeDataset.data")
print("Autogenerated Crime Model:")
autoCrimeModel = LinearRegressionModel(dataset=autoCrimeDataset,
                                       predictor_variables=[],
                                       response_variables=["ViolentCrimesPerPop"],
                                       test_size_percentage=0.1,
                                       nan_val="?",
                                       target_accuracy=.9,
                                       auto_select_predictors=True)

airfoilNoiseDataset = pd.read_table("../Datasets/airfoilNoiseDataset.dat")
print("Airfoil Noise Model:")
airfoilModel = LinearRegressionModel(dataset=airfoilNoiseDataset,
                                     predictor_variables=["Frequency", "AngleAttack", "ChordLength", "FreeStreamVel",
                                                          "SuctionSideDisplacementThickness"],
                                     response_variables=["SoundPressureDB"],
                                     test_size_percentage=0.01,
                                     target_accuracy=.90)

autoAirfoilNoiseDataset = pd.read_table("../Datasets/airfoilNoiseDataset.dat")
print("Autogenerated Airfoil Noise Model:")
autoAirfoilModel = LinearRegressionModel(dataset=autoAirfoilNoiseDataset,
                                         predictor_variables=[],
                                         response_variables=["SoundPressureDB"],
                                         test_size_percentage=0.01,
                                         target_accuracy=.5,
                                         auto_select_predictors=True)

studentDataset = pd.read_csv("../Datasets/student-mat.csv", sep=";")
print("Student Model:")
studentModel = LinearRegressionModel(dataset=studentDataset,
                                     predictor_variables=["studytime", "absences", "failures", "G1", "G2"],
                                     response_variables=["G3"],
                                     test_size_percentage=0.1,
                                     target_accuracy=.9)

autoStudentDataset = pd.read_csv("../Datasets/student-mat.csv", sep=";")
print("Autogenerated Student Model:")
autoStudentModel = LinearRegressionModel(dataset=autoStudentDataset,
                                         predictor_variables=[],
                                         response_variables=["G3"],
                                         test_size_percentage=0.1,
                                         target_accuracy=.9,
                                         auto_select_predictors=True)

singleVarDataset = pd.read_csv("../Datasets/single_var_lin_reg.csv")
print("Single Variable Model:")
singleVarModel = LinearRegressionModel(dataset=singleVarDataset,
                                       predictor_variables=["x"],
                                       response_variables=["y"],
                                       test_size_percentage=0,
                                       target_accuracy=.95)
print("Single Variable Model Test Data:")
singleVarTestDataset = pd.read_csv("../Datasets/single_var_lin_reg_test.csv")
print("Accuracy: " + str(singleVarModel.score(singleVarTestDataset["x"], singleVarTestDataset["y"])))
# PlotUtil.plot(x_data=singleVarTestDataset["x"], y_data=singleVarTestDataset["y"])
