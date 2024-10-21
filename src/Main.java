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
       } catch (Exception e) {
           e.printStackTrace();
        }
    }
}
