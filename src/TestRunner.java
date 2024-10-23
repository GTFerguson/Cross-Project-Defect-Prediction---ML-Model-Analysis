import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.util.*;

public class TestRunner {
    private static final int NUM_OF_FOLDS = 10; // Cross-Fold Validation
    private static final int SEED = 42;

    // Storage for evaluations produced by tests. Each
    private static Map<String, List<EvaluationResult>> eval_map = new HashMap<String, List<EvaluationResult>>();

    // Runs a single test on a given dataset using the provided model
    public Evaluation run_test (Map.Entry<String, AbstractClassifier> model_entry,
                                Map.Entry<String, Instances> testing_set_entry) throws Exception {
            System.out.println("Testing '"+ model_entry.getKey() + "' on dataset '" + testing_set_entry.getKey() + "'");
            Instances current_dataset = testing_set_entry.getValue();
            Evaluation eval = new Evaluation(testing_set_entry.getValue());
            eval.crossValidateModel(
                    model_entry.getValue(), testing_set_entry.getValue(), NUM_OF_FOLDS, new Random(SEED)
            );
            return eval;
    }

    public boolean train_model (Map.Entry<String, AbstractClassifier> model_entry,
                                Map.Entry<String, Instances> training_set_entry) {
        try {
            model_entry.getValue().buildClassifier(training_set_entry.getValue());
            System.out.println("Successfully trained model '" + model_entry.getKey() + "' on dataset '" + training_set_entry.getKey() + "'!");
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Could not train model '" + model_entry.getKey() + "' on dataset '" + training_set_entry.getKey() + "'");
            return false;
        }
        return true;
    }

    // Helper method to run batch testing on a given Map of datasets.
    public void run_tests (Map.Entry<String, AbstractClassifier> model_entry,
                           Map<String, Instances> datasets, String training_set_name) {
        // Results of evaluations are stored here, first checking if an entry already exists
        List<EvaluationResult> evaluations = eval_map.getOrDefault(model_entry.getKey(), new ArrayList<>());

        for (Map.Entry<String, Instances> testing_set_entry : datasets.entrySet()) {
            try {
                Evaluation eval = run_test(model_entry, testing_set_entry);
                evaluations.add(new EvaluationResult(training_set_name, testing_set_entry.getKey(), eval));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        eval_map.put(model_entry.getKey(), evaluations);
    }

    // This is a cross-project defect prediction test. Meaning a model is trained on one dataset and then evaluated
    // on another.
    public void run_cpdp_test (ModelHandler model_handler, Map<String, Instances> datasets) {
        for (Map.Entry<String, AbstractClassifier> model_entry: model_handler.get_model_map().entrySet()) {
            // Iterate training on each dataset
            for (Map.Entry<String, Instances> training_set_entry : datasets.entrySet()) {
                if (train_model(model_entry, training_set_entry)) {
                    run_tests(model_entry, datasets, training_set_entry.getKey());
                }
            }
        }
    }

    public String evaluation_results_to_string () {
        StringBuilder output = new StringBuilder();
        // Print header for the table
        output.append(
                String.format("%-25s %-20s %-20s %-10s %-10s %-10s\n",
                "Model Name", "Training Set", "Testing Set", "Accuracy", "Recall", "F-Measure")
        );
        output.append("---------------------------------------------------------------------------------------------------\n");

        for (Map.Entry<String, List<EvaluationResult>> entry : eval_map.entrySet()) {
            String model_name = entry.getKey().trim();
            for (EvaluationResult result : entry.getValue()) {
                Evaluation eval = result.get_evaluation();
                // Print metrics in a table row
                output.append(String.format("%-25s %-20s %-20s %-10.4f %-10.4f %-10.4f\n",
                        model_name, result.get_training_set_name(), result.get_testing_set_name(),
                        eval.pctCorrect() / 100, eval.recall(1), eval.fMeasure(1))
                );
            }
        }
        return output.toString();
    }

    public String summarise_results () {
        StringBuilder summary = new StringBuilder();

        // Print header for the table
        summary.append(
                String.format("%-25s %-10s %-10s %-10s\n",
                        "Model Name", "Accuracy", "Recall", "F-Measure")
        );
        summary.append("---------------------------------------------------------\n");

        for (Map.Entry<String, List<EvaluationResult>> entry : eval_map.entrySet()) {
            String model_name = entry.getKey();
            List<EvaluationResult> results = entry.getValue();

            // Variables to accumulate the metrics
            double accuracy = 0.0;
            double recall = 0.0;
            double f_measure = 0.0;
            int eval_count = results.size();

            // Sum up the metrics
            for (EvaluationResult result : results) {
                Evaluation eval = result.get_evaluation();

                // Check if the test and training set are the same...
                if (result.get_testing_set_name().equals(result.get_training_set_name())) {
                    // if so we don't include it in the summary and reduce the eval count
                    --eval_count;
                } else {
                    // otherwise, tally the metrics
                    accuracy += eval.pctCorrect() / 100;
                    recall += eval.recall(1);
                    // If recall or precision is 0, f-measure will return NaN
                    // This can be assumed as 0 and therefore removed from final results
                    if (!Double.isNaN(eval.fMeasure(1))) f_measure += eval.fMeasure(1);
                }
            }

            // Calculate the averages
            accuracy    /= eval_count;
            recall      /= eval_count;
            f_measure   /= eval_count;

            summary.append(String.format("%-25s %-10.4f %-10.4f %-10.4f\n",
                    model_name, accuracy, recall, f_measure)
            );
        }
        return summary.toString();
    }

    public String summarise_results_per_training_set () {
        StringBuilder summary = new StringBuilder();

        // Print header for the summary table
        summary.append(String.format("%-25s %-20s %-10s %-10s %-10s %-10s\n",
                "Model Name", "Training Set", "Count", "Accuracy", "Recall", "F-Measure"));
        summary.append("-------------------------------------------------------------------------------\n");

        Map<String, Map<String, List<EvaluationResult>>> sorted_evals = new HashMap<>();
        // Iterate through the eval_map to get model names and their evaluations
        for (Map.Entry<String, List<EvaluationResult>> entry : eval_map.entrySet()) {
            String model_name = entry.getKey();

            // Group results by training set name
            for (EvaluationResult eval_result : entry.getValue()) {
                String training_set_name = eval_result.get_training_set_name();

                // This complicated mess checks retrieves existing record if it exists and adds eval_result
                sorted_evals
                        .computeIfAbsent(model_name, k -> new HashMap<>())
                        .computeIfAbsent(training_set_name, k -> new ArrayList<>()).add(eval_result);
            }
        }

        // For each model...
        for (Map.Entry<String, Map<String, List<EvaluationResult>>> model_results_entry : sorted_evals.entrySet()) {
            String model_name = model_results_entry.getKey();
            Map<String, List<EvaluationResult>> model_results = model_results_entry.getValue();

            // for each training set...
            for (Map.Entry<String, List<EvaluationResult>> training_set_results : model_results.entrySet()) {
                List<EvaluationResult> eval_list = training_set_results.getValue();
                int eval_count = eval_list.size();

                // Variables to store our tallies
                double accuracy     = 0.0;
                double recall       = 0.0;
                double f_measure    = 0.0;

                // Iterate over all EvaluationResult objects to calculate totals
                for (EvaluationResult eval_result : eval_list) {
                    Evaluation eval = eval_result.get_evaluation();

                    accuracy    += eval.pctCorrect() / 100;
                    recall      += eval.recall(1);
                    if (!Double.isNaN(eval.fMeasure(1))) f_measure += eval.fMeasure(1);
                }

                // Calculate averages
                accuracy    /= eval_count;
                recall      /= eval_count;
                f_measure   /= eval_count;

                // Print metrics for the current model and training set
                summary.append(String.format("%-25s %-20s %-10d %-10.4f %-10.4f %-10.4f\n",
                        model_name, training_set_results.getKey(), eval_count, accuracy, recall, f_measure));
            }
        }
        return summary.toString();
    }

}