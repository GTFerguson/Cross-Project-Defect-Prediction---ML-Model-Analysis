import weka.classifiers.immune.immunos.Immunos2;
import weka.core.Instances;
import weka.core.WekaPackageManager;
import weka.core.Utils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import java.util.Random;

public class Main {

    public static void load_nasa_datasets (DatasetLoader _loader) {
       _loader.load_dataset("CM1", "datasets/promise/cm1.arff");
       _loader.load_dataset("JM1", "datasets/promise/jm1.arff");
       _loader.load_dataset("KC1", "datasets/promise/kc1.arff");
       _loader.load_dataset("KC2", "datasets/promise/kc1.arff");
       _loader.load_dataset("PC1", "datasets/promise/pc1.arff");
    }

    public static void run_test (DatasetLoader _loader, Immunos2 _immunos2) {
        try {
            Instances current_dataset = _loader.get_dataset("CM1");
            Evaluation eval = new Evaluation(current_dataset);
            eval.crossValidateModel(_immunos2, current_dataset, 10, new Random(42));
            System.out.println(_immunos2.toString());
            System.out.println(eval.toSummaryString());
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static void main(String[] args) {
        try {
            WekaPackageManager.loadPackages(false);
            DatasetLoader loader = new DatasetLoader();
            load_nasa_datasets(loader);
            Immunos2 immunos2 = new Immunos2();
            //run_test(loader, immunos2);
       } catch (Exception e) {
           e.printStackTrace();
        }
    }
}
