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
    private static Map<String, Evaluation> eval_map = new HashMap<String, Evaluation>();

    public static void run_test (DatasetLoader _loader, AbstractClassifier model) {
        try {
            System.out.println("Testing "+ model.toString());
            Instances current_dataset = _loader.get_dataset("CM1");
            Evaluation eval = new Evaluation(current_dataset);
            eval.crossValidateModel(model, current_dataset, 10, new Random(42));
            eval_map.put(model.toString(), eval);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

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

    public static void main(String[] args) {
        try {
            WekaPackageManager.loadPackages(false);
            DatasetLoader loader = new DatasetLoader();
            ModelHandler model_handler = new ModelHandler();

            model_handler.load_models();
            loader.load_nasa_datasets();

            // Loop over each model in the model handler map
            for (Map.Entry<String, AbstractClassifier> entry : model_handler.get_model_map().entrySet()) {
                String model_name = entry.getKey();
                AbstractClassifier model = entry.getValue();
                run_test(loader, model);
            }
            System.out.println(evaluations_to_string());

       } catch (Exception e) {
           e.printStackTrace();
        }
    }
}
