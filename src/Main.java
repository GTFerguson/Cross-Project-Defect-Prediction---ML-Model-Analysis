import weka.attributeSelection.AttributeSelection;
import weka.core.WekaPackageManager;
import java.util.Scanner;

public class Main {

    public static void run_tests (ModelHandler model_handler, DatasetLoader dataset_loader, TestRunner test_runner) {
        try {
            test_runner.run_cpdp_test(model_handler, dataset_loader.get_datasets());
            System.out.println("Test Evaluations");
            System.out.println(test_runner.evaluation_results_to_string());
            System.out.println("Test Summarisations");
            System.out.println(test_runner.summarise_results_per_training_set());
            System.out.println();
            System.out.println(test_runner.summarise_results());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main_menu (ModelHandler model_handler, DatasetLoader dataset_loader, TestRunner test_runner,
                                  FeatureSelection feature_selection) {
        Scanner scanner = new Scanner(System.in);

//        System.out.println("Preprocess missing values? (Y/N)");
//        String user_input = scanner.nextLine().trim().toLowerCase();
        //if (user_input.equals("y"))
        dataset_loader.preprocess_datasets();

        run_tests(model_handler, dataset_loader, test_runner);

    }

    public static void main(String[] args) {
        // Setup our objects
        EvaluationsDB eval_db = new EvaluationsDB();
        WekaPackageManager.loadPackages(false);
        ModelHandler model_handler = new ModelHandler();
        DatasetLoader dataset_loader = new DatasetLoader();
        TestRunner test_runner = new TestRunner();
        FeatureSelection feature_selection = new FeatureSelection();

        eval_db.connect();

        main_menu(model_handler, dataset_loader, test_runner, feature_selection);
    }
}
