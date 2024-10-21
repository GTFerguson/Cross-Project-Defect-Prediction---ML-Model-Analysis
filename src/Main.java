import weka.classifiers.AbstractClassifier;
import weka.classifiers.immune.immunos.Immunos2;
import weka.core.Instances;
import weka.core.WekaPackageManager;
import weka.core.Utils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import java.util.Random;

public class Main {
    public static void run_test (DatasetLoader _loader, AbstractClassifier model) {
        try {
            System.out.println("Test Starting");
            Instances current_dataset = _loader.get_dataset("CM1");
            Evaluation eval = new Evaluation(current_dataset);
            eval.crossValidateModel(model, current_dataset, 10, new Random(42));
            System.out.println(model.toString());
            System.out.println(eval.toSummaryString());
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

            run_test(loader, model_handler.get_model("Immunos-1"));
       } catch (Exception e) {
           e.printStackTrace();
        }
    }
}
