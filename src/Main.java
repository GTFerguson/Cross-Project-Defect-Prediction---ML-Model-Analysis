import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.core.WekaPackageManager;
import weka.core.Utils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class Main {

    /*
    public static String evaluations_to_string () {
        StringBuilder output = new StringBuilder();
        // Print header for the table
        output.append(String.format("%-30s %-10s %-10s %-10s\n", "Model Name", "Accuracy", "Recall", "F-Measure"));
        output.append("----------------------------------------------------------\n");

        for (Map.Entry<String, Evaluation> entry : eval_map.entrySet()) {
            Evaluation eval = entry.getValue();
            // Print metrics in a table row
            output.append(String.format("%-30s %-10.4f %-10.4f %-10.4f\n",
                    entry.getKey().trim(), eval.pctCorrect()/100, eval.recall(1), eval.fMeasure(1)));
        }
        return output.toString();
    }
     */

    public static void main(String[] args) {
        try {
            WekaPackageManager.loadPackages(false);
            DatasetLoader loader = new DatasetLoader();
            ModelHandler model_handler = new ModelHandler();
            TestRunner test_runner = new TestRunner();

            model_handler.load_models();
            loader.load_nasa_datasets();

            test_runner.run_cpdp_test(model_handler, loader.get_datasets(), "CM1");
            System.out.println("Test Evaluations");
            System.out.println(test_runner.evaluation_results_to_string());
            System.out.println("Test Summarisations");
            System.out.println(test_runner.summarise_results());

       } catch (Exception e) {
           e.printStackTrace();
        }
    }
}
