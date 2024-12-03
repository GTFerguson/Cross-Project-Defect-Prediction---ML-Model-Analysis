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

    public boolean train_model (Map.Entry<String, AbstractClassifier> model_entry,
                                Instances training_set, String training_set_key) {
        try {
            model_entry.getValue().buildClassifier(training_set);
            System.out.println("Successfully trained model '" + model_entry.getKey() + "' on dataset '" + training_set_key + "'!");
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Could not train model '" + model_entry.getKey() + "' on dataset '" + training_set_key + "'");
            return false;
        }
        return true;
    }

    // Runs a single test on a given dataset using the provided model
    public Evaluation run_test (Map.Entry<String, AbstractClassifier> model_entry,
                                String testing_set_name, Instances testing_set) throws Exception {
            System.out.println("Testing '"+ model_entry.getKey() + "' on dataset '" + testing_set_name + "'");
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
    public List<EvaluationResult> run_tests (Map.Entry<String, AbstractClassifier> model_entry,
                           Map<String, Instances> datasets,
                           Map.Entry<String, Instances> training_set) {

        // Results of evaluations are stored here, first checking if an entry already exists
//        List<EvaluationResult> evaluations = eval_map.getOrDefault(model_entry.getKey(), new ArrayList<>());
        List<EvaluationResult> evaluations = new ArrayList<>();

        for (Map.Entry<String, Instances> testing_set_entry : datasets.entrySet()) {
            // Run test and store evaluation data
            try {
                Evaluation eval = run_test(model_entry, testing_set_entry.getKey(), testing_set_entry.getValue());
                evaluations.add(EvaluationResult.create(model_entry.getKey(), training_set.getKey(), testing_set_entry.getKey(), eval));
            } catch (Exception e) {
                System.out.println("Test could not be performed.");
                e.printStackTrace();
            }
        }
        return evaluations;
    }

    // Overloaded for feature selection
    // Helper method to run batch testing on a given Map of datasets.
    public List<EvaluationResult> run_tests (Map.Entry<String, AbstractClassifier> model_entry,
                           Map<String, Instances> datasets,
                           String training_set_name,
                           int[] selected_attributes) {

        // Results of evaluations are stored here, first checking if an entry already exists
//        List<EvaluationResult> evaluations = eval_map.getOrDefault(model_entry.getKey(), new ArrayList<>());
        List<EvaluationResult> evaluations = new ArrayList<>();

        for (Map.Entry<String, Instances> testing_set_entry : datasets.entrySet()) {
            Instances testing_set = testing_set_entry.getValue();
            Instances reduced_testing_set = testing_set;

            // Remove all attributes not selected by the training selector
            for (int i = testing_set.numAttributes() - 1; i >= 0; i--) {
                if (!array_contains(selected_attributes, i)) {
                    reduced_testing_set.deleteAttributeAt(i);
                }
            }

            // Run test and store evaluation data
            try {
                Evaluation eval = run_test(model_entry, testing_set_entry.getKey(), reduced_testing_set);
                evaluations.add(EvaluationResult.create(model_entry.getKey(), training_set_name, testing_set_entry.getKey(), eval));
            } catch (Exception e) {
                System.out.println("Test could not be performed.");
                System.out.println("Original testing set attributes: " + testing_set_entry.getValue().numAttributes());
                System.out.println("Reduced testing set attributes: " + reduced_testing_set.numAttributes());
                System.out.println("Testing set class index: " + reduced_testing_set.classIndex());
                e.printStackTrace();
            }
        }
        return evaluations;
//        eval_map.put(model_entry.getKey(), evaluations);
    }

//    // This is a cross-project defect prediction test. Meaning a model is trained on one dataset and then evaluated
//    // on another.
//    public void run_cpdp_test (ModelHandler model_handler, Map<String, Instances> datasets) throws Exception {
//        // For each model provided
//        for (Map.Entry<String, AbstractClassifier> model_entry: model_handler.get_model_map().entrySet()) {
//            // Iterate training on each dataset
//            for (Map.Entry<String, Instances> training_set_entry : datasets.entrySet()) {
//                // Now attempt to train model and run tests if successful
//                if (train_model(model_entry, training_set_entry.getValue(), training_set_entry.getKey())) {
//                    run_tests(model_entry, datasets, training_set_entry);
//                }
//            }
//        }
//    }

    // Overloaded version to allow for feature selection
    public List<EvaluationResult> run_cpdp_test (ModelHandler model_handler, Map<String, Instances> datasets,
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
                    if (train_model(model_entry, training_set_entry.getValue(), training_set_entry.getKey())) {
                        List<EvaluationResult> evaluations = run_tests(model_entry, datasets, training_set_entry);
                        master_eval_list.addAll(evaluations);
                    }

                // Feature Selection
                } else {
                    for (Double threshold = start_threshold; threshold <= end_threshold; threshold += step) {
                        System.out.printf("Testing with threshold: %.2f%n", threshold);
                        AttributeSelection selector = feature_selection.train_selector
                                (evaluator, search_method, training_set_entry.getValue(), threshold);
                        Instances reduced_training_set = selector.reduceDimensionality(training_set_entry.getValue());

                        // Get the attributes that have been selected so they can be applied to the test sets
                        int[] selected_attributes = selector.selectedAttributes();

                        // Now attempt to train model and run tests if successful
                        if (train_model(model_entry, reduced_training_set, training_set_entry.getKey())) {
                            List<EvaluationResult> evaluations = run_tests(model_entry, datasets, training_set_entry.getKey(), selected_attributes);
                            // Add metadata and combine results
                            for (EvaluationResult result : evaluations) {
                                result.set_evaluator(evaluator);
                                result.set_search_method(search_method);
                                result.set_threshold(threshold);
                                master_eval_list.add(result);
                            }
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