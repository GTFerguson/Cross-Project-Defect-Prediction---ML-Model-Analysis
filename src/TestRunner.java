import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.util.*;

// Weka's Evaluation class does not allow for storage of the training and test dataset names, so we must create our
// own custom class for this purpose.
class EvaluationResult {
    private String training_set_name;
    private String testing_set_name;
    private Evaluation evaluation;

    public EvaluationResult(String training_set_name, String testing_set_name, Evaluation evaluation) {
        this.training_set_name = training_set_name;
        this.testing_set_name = testing_set_name;
        this.evaluation = evaluation;
    }

    public String get_training_set_name() {
        return training_set_name;
    }

    public String get_testing_set_name() {
        return testing_set_name;
    }

    public Evaluation get_evaluation() {
        return evaluation;
    }
}

public class TestRunner {
    // Storage for evaluations produced by tests. Each
    private static Map<String, List<EvaluationResult>> eval_map = new HashMap<String, List<EvaluationResult>>();

    public Evaluation run_test (AbstractClassifier model, Instances testing_data) throws Exception {
            System.out.println("Testing '"+ model.toString());
            Instances current_dataset = testing_data;
            Evaluation eval = new Evaluation(testing_data);
            eval.crossValidateModel(model, testing_data, 10, new Random(42));
            return eval;
    }

    public boolean train_model (AbstractClassifier model, Instances training_set) {
        try {
            model.buildClassifier(training_set);
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
        return true;
    }

    // This is a cross-project defect prediction test. Meaning a model is trained on one dataset and then evaluated
    // on another.
    public void run_cpdp_test (ModelHandler model_handler, Map<String, Instances> datasets, String training_set_name) {
        for (Map.Entry<String, AbstractClassifier> model_entry: model_handler.get_model_map().entrySet()) {
            String model_name = model_entry.getKey();
            AbstractClassifier model = model_entry.getValue();

            if (!train_model(model, datasets.get(training_set_name))) {
                System.out.println("Could not train model '" + model.toString() + "' on dataset '" + training_set_name + "'");
                System.out.println("CPDP test aborted!");
            } else {
                List<EvaluationResult> evaluations = new ArrayList<>();
                // If model successfully trained we can run our tests
                for (Map.Entry<String, Instances> dataset_entry : datasets.entrySet()) {
                    // Create an empty list to store the evaluation results for this model
                    try {
                        Evaluation eval = run_test(model, dataset_entry.getValue());
                        evaluations.add(new EvaluationResult(training_set_name, dataset_entry.getKey(), eval));
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
                eval_map.put(model_name, evaluations);
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
}
