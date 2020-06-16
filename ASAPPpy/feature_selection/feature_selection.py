'''
Module used for feature selection.
'''

import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectPercentile, SelectFromModel, RFE, f_regression
from ASAPPpy.scripts.tools import write_data_to_file

def build_mask(mask, used_features):
	""" Function used to complete the mask with unused features not available for feature selection """

	for index, key in enumerate(used_features):
		if not mask[index]:
			used_features[key] = 0

	return used_features

def best_percentile_selector(train_features, test_features, train_similarity_target, test_similarity_target, regressor):
	""" Function used to select the best percentile selector """
	percentile_score = 0
	percentiles = [25, 35, 45, 50, 55, 65, 75]
	# percentiles = [45]
	percentile_selector = None
	percentile_train_features_selected = None
	percentile_test_features_selected = None

	for percentile in percentiles:
		print(percentile)
		temp_percentile_selector = SelectPercentile(score_func=f_regression, percentile=percentile)
		temp_percentile_selector.fit(train_features, train_similarity_target)
		temp_percentile_train_features_selected = temp_percentile_selector.transform(train_features)
		temp_percentile_test_features_selected = temp_percentile_selector.transform(test_features)

		regressor.fit(temp_percentile_train_features_selected, train_similarity_target)

		temp_score = regressor.score(temp_percentile_test_features_selected, test_similarity_target)
		print("The score on the selected features (Percentile Selector): %.3f" % temp_score)

		if temp_score > percentile_score:
			percentile_score = temp_score
			percentile_selector = temp_percentile_selector
			percentile_train_features_selected = temp_percentile_train_features_selected
			percentile_test_features_selected = temp_percentile_test_features_selected

	percentile_mask = percentile_selector.get_support()
	print("This is the percentile mask: ")
	print(percentile_mask)

	return percentile_selector, percentile_score, percentile_train_features_selected, percentile_test_features_selected, percentile_mask

def best_model_based_selector(train_features, test_features, train_similarity_target, test_similarity_target, regressor):
	""" Function used to select the best model based selector """
	model_based_score = 0
	scaling_factors = ["0.25*mean", "0.5*mean", "median", "1.25*mean", "1.5*mean"]
	# scaling_factors = ["0.5*mean", "median"]
	model_based_selector = None
	model_based_train_features_selected = None
	model_based_test_features_selected = None

	for factor in scaling_factors:
		print(factor)
		temp_model_based_selector = SelectFromModel(RandomForestRegressor(n_estimators=100), threshold=factor)
		temp_model_based_selector.fit(train_features, train_similarity_target)
		temp_model_based_train_features_selected = temp_model_based_selector.transform(train_features)
		temp_model_based_test_features_selected = temp_model_based_selector.transform(test_features)

		regressor.fit(temp_model_based_train_features_selected, train_similarity_target)

		temp_score = regressor.score(temp_model_based_test_features_selected, test_similarity_target)
		print("The score on the selected features (Model Based Selector): %.3f" % temp_score)

		if temp_score > model_based_score:
			model_based_score = temp_score
			model_based_selector = temp_model_based_selector
			model_based_train_features_selected = temp_model_based_train_features_selected
			model_based_test_features_selected = temp_model_based_test_features_selected

	model_based_mask = model_based_selector.get_support()
	print("This is the model based mask: ")
	print(model_based_mask)

	return model_based_selector, model_based_score, model_based_train_features_selected, model_based_test_features_selected, model_based_mask

def best_iterative_based_selector(train_features, test_features, train_similarity_target, test_similarity_target, regressor):
	""" Function used to select the best iterative based selector """
	iterative_based_score = 0
	# given that all pairs use the same amount of features, the position 0 was arbitrarily selected to compute the number of features being used
	min_number_features = int(0.15*len(train_features[0]))
	max_number_features = int(0.85*len(train_features[0]))

	# min_number_features = 19
	# max_number_features = 20

	iterative_based_selector = None
	iterative_based_train_features_selected = None
	iterative_based_test_features_selected = None

	for i in range(min_number_features, max_number_features):
		print(i)
		temp_iterative_based_selector = RFE(RandomForestRegressor(n_estimators=100), n_features_to_select=i)
		temp_iterative_based_selector.fit(train_features, train_similarity_target)
		temp_iterative_based_train_features_selected = temp_iterative_based_selector.transform(train_features)
		temp_iterative_based_test_features_selected = temp_iterative_based_selector.transform(test_features)

		regressor.fit(temp_iterative_based_train_features_selected, train_similarity_target)

		temp_score = regressor.score(temp_iterative_based_test_features_selected, test_similarity_target)
		print("The score on the selected features (Iterative Based Selector): %.3f" % temp_score)

		if temp_score > iterative_based_score:
			iterative_based_score = temp_score
			iterative_based_selector = temp_iterative_based_selector
			iterative_based_train_features_selected = temp_iterative_based_train_features_selected
			iterative_based_test_features_selected = temp_iterative_based_test_features_selected

	iterative_based_mask = iterative_based_selector.get_support()
	print("This is the iterative based mask: ")
	print(iterative_based_mask)

	return iterative_based_selector, iterative_based_score, iterative_based_train_features_selected, iterative_based_test_features_selected, iterative_based_mask

def feature_selection(train_features, test_features, train_similarity_target, test_similarity_target, regressor, used_features):
	# TODO: Check why were test_features being returned
	""" Function used to perform feature selection """
	# percentile selector
	percentile_selector, percentile_score, percentile_train_features_selected, percentile_test_features_selected, percentile_mask = best_percentile_selector(train_features, test_features, train_similarity_target, test_similarity_target, regressor)

	# model based selector
	model_based_selector, model_based_score, model_based_train_features_selected, model_based_test_features_selected, model_based_mask = best_model_based_selector(train_features, test_features, train_similarity_target, test_similarity_target, regressor)

	# iterative based selector
	iterative_based_selector, iterative_based_score, iterative_based_train_features_selected, iterative_based_test_features_selected, iterative_based_mask = best_iterative_based_selector(train_features, test_features, train_similarity_target, test_similarity_target, regressor)

	all_scores = []

	regressor.fit(train_features, train_similarity_target)
	print("The score on all features: %.3f" % regressor.score(test_features, test_similarity_target))
	all_scores.append(regressor.score(test_features, test_similarity_target))

	# show results for the percentile selector
	all_scores.append(percentile_score)

	# show results for the model based selector
	all_scores.append(model_based_score)

	# show results for the iterative based selector
	all_scores.append(iterative_based_score)

	max_value_position = all_scores.index(max(all_scores))

	if max_value_position == 0:
		print("Returning all features!\n")
		return None
	elif max_value_position == 1:
		percentile_mask = build_mask(percentile_mask, used_features)
		print(percentile_mask)

		print("Returning features selected with the percentile selector!\n")
		return percentile_selector, percentile_train_features_selected, percentile_mask
	elif max_value_position == 2:
		model_based_mask = build_mask(model_based_mask, used_features)
		print(model_based_mask)

		print("Returning features selected with the model based selector!\n")
		return model_based_selector, model_based_train_features_selected, model_based_mask
	else:
		iterative_based_mask = build_mask(iterative_based_mask, used_features)
		print(iterative_based_mask)

		print("Returning features selected with the iterative based selector!\n")
		return iterative_based_selector, iterative_based_train_features_selected, iterative_based_mask
