import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.Ranker;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.util.*;

public class TestRunner {
    private static final int NUM_OF_FOLDS = 10; // Cross-Fold Validation
    private static final int SEED = 42;

    // Storage for evaluations produced by tests.
    private Map<String, List<EvaluationResult>> eval_map = new HashMap<String, List<EvaluationResult>>();

    public boolean train_model (String model_name, AbstractClassifier model,
                                String training_set_name, Instances training_set) {
        try {
            model.buildClassifier(training_set);
            System.out.println("Successfully trained model '" + model_name + "' on dataset '" + training_set_name + "'!");
        } catch (Exception e) {
            System.out.println("Could not train model '" + model_name + "' on dataset '" + training_set_name + "'");
            e.printStackTrace();
            return false;
        }
        return true;
    }

    // Convenience method to pass Map entries
    public boolean train_model (Map.Entry<String, AbstractClassifier> model_entry,
                                Map.Entry<String, Instances> training_set_entry) {
        return train_model(
                model_entry.getKey(), model_entry.getValue(),
                training_set_entry.getKey(), training_set_entry.getValue()
        );
    }

    // Convenience method to pass Map entries
    public boolean train_model (Map.Entry<String, AbstractClassifier> model_entry,
                                String training_set_name, Instances training_set) {
        return train_model(
                model_entry.getKey(), model_entry.getValue(),
                training_set_name, training_set
        );
    }
    // Runs a single test on a given dataset using the provided model
    public Evaluation run_test (Map.Entry<String, AbstractClassifier> model_entry,
                                String testing_set_name, Instances testing_set) throws Exception {
            System.out.printf("Testing '%s' on dataset '%s'%n", model_entry.getKey(), testing_set_name);
            Instances current_dataset = testing_set;
            Evaluation eval = new Evaluation(testing_set);
            eval.evaluateModel(model_entry.getValue(), testing_set);
            return eval;
    }

    // Helper method to check if an array contains an element
    private boolean array_contains(int[] array, int value) {
        for (int i : array) {
            if (i == value) {
                return true;
            }
        }
        return false;
    }

    // Helper method to run batch testing on a given Map of datasets.
    public List<EvaluationResult> run_tests (
            Map.Entry<String, AbstractClassifier> model_entry,
            Map<String, Instances> datasets,
            String training_set_name,
            int[] selected_attributes) {

        // Results of evaluations are stored here, first checking if an entry already exists
        List<EvaluationResult> evaluations = new ArrayList<>();

        for (Map.Entry<String, Instances> testing_set_entry : datasets.entrySet()) {
            // Deep copy to ensure no changes are made to the original datasets
            Instances reduced_testing_set = new Instances(testing_set_entry.getValue());

            // If feature selection is applied remove all attributes not selected by the training selector
            if (selected_attributes != null) {
                for (int i = reduced_testing_set.numAttributes() - 1; i >= 0; i--) {
                    if (!array_contains(selected_attributes, i)) {
                        reduced_testing_set.deleteAttributeAt(i);
                    }
                }
                System.out.printf("Reduced attributes of '%s' to %d%n", training_set_name, reduced_testing_set.numAttributes());
            }

            // Run test and store evaluation data
            try {
                Evaluation eval = run_test(model_entry, testing_set_entry.getKey(), reduced_testing_set);
                evaluations.add(EvaluationResult.create(model_entry.getKey(), training_set_name, testing_set_entry.getKey(), eval));
            } catch (Exception e) {
                System.out.println("Test could not be performed.");
                if (selected_attributes != null) {
                    System.out.println("Original testing set attributes: " + testing_set_entry.getValue().numAttributes());
                    System.out.println("Reduced testing set attributes: " + reduced_testing_set.numAttributes());
                    System.out.println("Testing set class index: " + reduced_testing_set.classIndex());
                }
                e.printStackTrace();
            }
        }
        return evaluations;
    }

    // This is a helper method that trains a given model on the datasets provided, as well as testing it on them
    private List<EvaluationResult> perform_model_cpdp_tests (
            Map.Entry<String, AbstractClassifier> model_entry, String training_set_name, Instances training_set,
            Map<String, Instances> datasets, int[] selected_attributes,
            String evaluator, String search_method, Double threshold) throws Exception {

        List<EvaluationResult> model_evaluations = new ArrayList<>();

        // Attempt to train the model, if successful run tests
        if (train_model(model_entry, training_set_name, training_set)) {
            List<EvaluationResult> evaluations = run_tests(
                    model_entry, datasets, training_set_name, selected_attributes
            );

            // Add metadata for feature selection if used
            if (selected_attributes != null) {
                for (EvaluationResult result : evaluations) {
                    result.set_evaluator(evaluator);
                    result.set_search_method(search_method);
                    System.out.printf("Setting evalulation to %s and search method to %s!!%n", evaluator, search_method);
                    if (threshold != null) result.set_threshold(threshold);
                }
            }
            model_evaluations.addAll(evaluations);
        }
        return model_evaluations;
    }

    // Convenience method to perform model test with feature selection but no set threshold values
    private List<EvaluationResult> perform_model_cpdp_tests (
            Map.Entry<String, AbstractClassifier> model_entry,
            String training_set_name, Instances training_set,
            Map<String, Instances> datasets, int[] selected_attributes,
            String evaluator, String search_method) throws Exception {
        return perform_model_cpdp_tests(
                model_entry, training_set_name, training_set, datasets, selected_attributes,
                evaluator, search_method, null
        );
    }

    // Convenience method to perform model test without feature selection
    private List<EvaluationResult> perform_model_cpdp_tests (
            Map.Entry<String, AbstractClassifier> model_entry,
            String training_set_name, Instances training_set,
            Map<String, Instances> datasets) throws Exception {
        return perform_model_cpdp_tests(
                model_entry, training_set_name, training_set, datasets,
                null, null, null, null
        );
    }

    // Overloaded version to allow for feature selection
    public List<EvaluationResult> run_cpdp_test (
            ModelHandler model_handler, Map<String, Instances> datasets,
            String evaluator, String search_method,
            Double start_threshold, Double end_threshold, Double step) throws Exception {
        FeatureSelection feature_selection = new FeatureSelection();
        List<EvaluationResult> master_eval_list = new ArrayList<>();

        // For each model provided
        for (Map.Entry<String, AbstractClassifier> model_entry: model_handler.get_model_map().entrySet()) {
            // Iterate training on each dataset
            for (Map.Entry<String, Instances> training_set_entry : datasets.entrySet()) {

                // No Feature Selection
                if (evaluator == null && search_method == null) {
                    // Now attempt to train model and run tests if successful
                    if (train_model(model_entry, training_set_entry)) {
                        List<EvaluationResult> evaluations = run_tests(
                                model_entry, datasets, training_set_entry.getKey(), null
                        );
                        master_eval_list.addAll(evaluations);
                    }

                // Feature Selection
                } else {
                    // For non-ranker search methods, such as the kind used by CFS
                    if (!search_method.equals("Ranker")) {
                        // Create a deep copy of the training set to avoid modifying the original
                        Instances training_set = new Instances(training_set_entry.getValue());

                        // Build the feature selector
                        AttributeSelection selector = feature_selection.train_selector(
                                evaluator, search_method, training_set
                        );
                        // Apply the selector to our training set
                        Instances reduced_training_set = selector.reduceDimensionality(training_set);

                        // Get the attributes that have been selected so they can be applied to the test sets
                        int[] selected_attributes = selector.selectedAttributes();

                        // Run the relevant tests on our model
                        List<EvaluationResult> model_evaluations = perform_model_cpdp_tests(
                                model_entry, training_set_entry.getKey(), reduced_training_set,
                                datasets, selected_attributes, evaluator, search_method
                        );

                        master_eval_list.addAll(model_evaluations);

                    } else { // Ranker search method require threshold values
                        for (Double threshold = start_threshold; threshold <= end_threshold; threshold += step) {
                            System.out.printf("Testing with threshold: %.2f%n", threshold);
                            AttributeSelection selector = feature_selection.train_selector (
                                    evaluator, search_method, training_set_entry.getValue(), threshold
                            );
                            Instances reduced_training_set = selector.reduceDimensionality(training_set_entry.getValue());

                            // Get the attributes that have been selected so they can be applied to the test sets
                            int[] selected_attributes = selector.selectedAttributes();

                            List<EvaluationResult> model_evaluations = perform_model_cpdp_tests(
                                    model_entry, training_set_entry.getKey(), reduced_training_set, datasets,
                                    selected_attributes, evaluator, search_method, threshold
                            );
                            master_eval_list.addAll(model_evaluations);
                        }
                    }
                }
            }
        }
        return master_eval_list;
    }

    // Convenience method for no feature selection
    public List<EvaluationResult> run_cpdp_test (ModelHandler model_handler, Map<String, Instances> datasets) throws Exception {
        return run_cpdp_test(model_handler, datasets, null, null, null);
    }

    // Convenience method for feature selection without threshold
    public List<EvaluationResult> run_cpdp_test (ModelHandler model_handler, Map<String, Instances> datasets,
                                                   String evaluator, String search_method) throws Exception {
        return run_cpdp_test(model_handler, datasets, evaluator, search_method, null);
    }

    // Convenience method for feature selection with single threshold use
    public List<EvaluationResult> run_cpdp_test (ModelHandler model_handler, Map<String, Instances> datasets,
                              String evaluator, String search_method, Double threshold) throws Exception {
        return run_cpdp_test(model_handler, datasets, evaluator, search_method, threshold, threshold, 1.0);
    }

    public String evaluation_results_to_string (List<EvaluationResult> evaluations) {
        StringBuilder output = new StringBuilder();
        // Print header for the table
        output.append(
                String.format("%-25s %-20s %-20s %-10s %-10s %-10s\n",
                "Model Name", "Training Set", "Testing Set", "Accuracy", "Recall", "F-Measure")
        );
        output.append("---------------------------------------------------------------------------------------------------\n");

        for (EvaluationResult evaluation : evaluations) {
            Evaluation eval = evaluation.get_evaluation();
            // Print metrics in a table row
            output.append(String.format("%-25s %-20s %-20s %-10.4f %-10.4f %-10.4f\n",
                    evaluation.get_training_set_name(), evaluation.get_training_set_name(),
                    evaluation.get_testing_set_name(),
                    eval.pctCorrect() / 100, eval.recall(1), eval.fMeasure(1))
            );
        }
        return output.toString();
    }

    public String summarise_results (List<EvaluationResult> evaluations) {
        StringBuilder summary = new StringBuilder();

        // Print header for the table
        summary.append(
                String.format("%-25s %-10s %-10s %-10s\n",
                        "Model Name", "Accuracy", "Recall", "F-Measure")
        );
        summary.append("---------------------------------------------------------\n");

        // Group evaluations by model name
        Map<String, List<EvaluationResult>> grouped_by_model = new HashMap<>();
        for (EvaluationResult evaluation : evaluations) {
            grouped_by_model.computeIfAbsent(evaluation.get_model_name(), k -> new ArrayList<>()).add(evaluation);
        }

        // Calculate averages for each model
        for (Map.Entry<String, List<EvaluationResult>> entry : grouped_by_model.entrySet()) {
            String model_name = entry.getKey();
            List<EvaluationResult> modelResults = entry.getValue();

            double accuracy = 0.0;
            double recall = 0.0;
            double f_measure = 0.0;
            int eval_count = modelResults.size();

            for (EvaluationResult result : modelResults) {
                Evaluation eval = result.get_evaluation();
                accuracy += eval.pctCorrect() / 100;
                recall += eval.recall(1);
                if (!Double.isNaN(eval.fMeasure(1))) {
                    f_measure += eval.fMeasure(1);
                }
            }

            // Calculate averages
            accuracy /= eval_count;
            recall /= eval_count;
            f_measure /= eval_count;

            summary.append(String.format("%-25s %-10.4f %-10.4f %-10.4f\n",
                    model_name, accuracy, recall, f_measure));
        }
        return summary.toString();
    }

    public String summarise_results_per_training_set(List<EvaluationResult> evaluations) {
        StringBuilder summary = new StringBuilder();

        // Print header for the summary table
        summary.append(String.format("%-25s %-20s %-10s %-10s %-10s %-10s\n",
                "Model Name", "Training Set", "Count", "Accuracy", "Recall", "F-Measure"));
        summary.append("-----------------------------------------------------------------------------------------\n");

        // Group results by model and training set
        Map<String, Map<String, List<EvaluationResult>>> grouped_evaluations = new HashMap<>();
        for (EvaluationResult result : evaluations) {
            grouped_evaluations
                    .computeIfAbsent(result.get_training_set_name(), k -> new HashMap<>())
                    .computeIfAbsent(result.get_testing_set_name(), k -> new ArrayList<>())
                    .add(result);
        }

        // Calculate averages for each training set
        for (Map.Entry<String, Map<String, List<EvaluationResult>>> model_entry : grouped_evaluations.entrySet()) {
            String model_name = model_entry.getKey();
            Map<String, List<EvaluationResult>> training_set_evals = model_entry.getValue();

            for (Map.Entry<String, List<EvaluationResult>> training_set_entry : training_set_evals.entrySet()) {
                String training_set_name = training_set_entry.getKey();
                List<EvaluationResult> eval_list = training_set_entry.getValue();

                int eval_count = eval_list.size();
                double accuracy = 0.0;
                double recall = 0.0;
                double f_measure = 0.0;

                for (EvaluationResult evaluation : eval_list) {
                    Evaluation eval = evaluation.get_evaluation();
                    accuracy += eval.pctCorrect() / 100;
                    recall += eval.recall(1);
                    if (!Double.isNaN(eval.fMeasure(1))) {
                        f_measure += eval.fMeasure(1);
                    }
                }

                // Calculate averages
                accuracy /= eval_count;
                recall /= eval_count;
                f_measure /= eval_count;

                summary.append(String.format("%-25s %-20s %-10d %-10.4f %-10.4f %-10.4f\n",
                        model_name, training_set_name, eval_count, accuracy, recall, f_measure));
            }
        }

        return summary.toString();
    }
}