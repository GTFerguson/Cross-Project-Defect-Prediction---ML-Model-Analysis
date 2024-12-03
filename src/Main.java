import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.WekaPackageManager;

import javax.xml.crypto.Data;
import java.sql.SQLException;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class Main {

    public static List<EvaluationResult> run_tests (ModelHandler model_handler, DatasetLoader dataset_loader) throws Exception {
        TestRunner test_runner = new TestRunner();
        List<EvaluationResult> evaluations = test_runner.run_cpdp_test(model_handler, dataset_loader.get_datasets());
//            System.out.println("Test Evaluations");
//            System.out.println(test_runner.evaluation_results_to_string(evaluations));
        System.out.println("\nTest Summarisations");
        System.out.println(test_runner.summarise_results_per_training_set(evaluations) + "\n");
        System.out.println(test_runner.summarise_results(evaluations));
        return evaluations;
    }

    public static List<EvaluationResult> run_tests (ModelHandler model_handler, DatasetLoader dataset_loader,
                                  String evaluator, String search_method) throws Exception {
        TestRunner test_runner = new TestRunner();
        String fs_details = "Evaluator: " + evaluator + " Search Method: " + search_method;
        System.out.println("Beginning test for " + fs_details);

        List<EvaluationResult> evaluations = test_runner.run_cpdp_test(model_handler, dataset_loader.get_datasets(), evaluator, search_method);
        System.out.println("Test Summarisations");
        System.out.println(test_runner.summarise_results(evaluations));
        return evaluations;
    }

    public static void prompt_user_save_to_db (List<EvaluationResult> evaluations, EvaluationsDB eval_db) {
        System.out.println("Test complete, add results to database? (Y/N)");
        Scanner scanner = new Scanner(System.in);
        String user_input = scanner.nextLine().trim().toLowerCase();
        if (user_input.equals("y")) {
            try {
                eval_db.insert_evaluations(evaluations);
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }

    public static void run_all_tests (ModelHandler model_handler, DatasetLoader dataset_loader, EvaluationsDB eval_db) {
        // No feature selection
        try {
            List<EvaluationResult> evaluations = run_tests(model_handler, dataset_loader);
            prompt_user_save_to_db(evaluations, eval_db);
        } catch (Exception e) {
            System.out.println("Could not perform test without feature selection");
            e.printStackTrace();
        }

        FeatureSelection feature_selection = new FeatureSelection();
        for (Map.Entry<String, ASEvaluation> evaluator_entry : feature_selection.get_evaluators().entrySet()) {
            for (Map.Entry<String, ASSearch> search_method_entry : feature_selection.get_search_methods().entrySet()) {
                String fs_details = "Evaluator: " + evaluator_entry.getKey() + " Search Method: " + search_method_entry.getKey();
                try {
                    List<EvaluationResult> evaluations = run_tests(model_handler, dataset_loader, evaluator_entry.getKey(), search_method_entry.getKey());
                    prompt_user_save_to_db(evaluations, eval_db);
                } catch (Exception e) {
                    System.out.println("Could not run test using " + fs_details);
                    e.printStackTrace();
                }
            }
        }
    }

    public static void test_menu (ModelHandler model_handler, DatasetLoader dataset_loader, EvaluationsDB eval_db)
            throws Exception {
        System.out.println("1: No Feature Selection");
        System.out.println("2: CFS");
        System.out.println("3: Info Gain");
        System.out.println("4: Gain Ratio");

        Scanner scanner = new Scanner(System.in);
        int user_input = scanner.nextInt();

        switch (user_input) {
            case 1:
                List<EvaluationResult> evaluations = run_tests(model_handler, dataset_loader);
                prompt_user_save_to_db(evaluations, eval_db);
                break;
            case 2:
                run_tests(model_handler, dataset_loader, "CFS Subset", "Best First");
                break;
            case 3:
                run_tests(model_handler, dataset_loader, "Info Gain", "Ranker");
                break;
            case 4:
                run_tests(model_handler, dataset_loader, "Gain Ratio", "Ranker");
                break;
            default:
                break;
        }

    }

    public static void main_menu (ModelHandler model_handler, DatasetLoader dataset_loader, EvaluationsDB eval_db) {
        Scanner scanner = new Scanner(System.in);

//        System.out.println("Preprocess missing values? (Y/N)");
//        String user_input = scanner.nextLine().trim().toLowerCase();
        //if (user_input.equals("y"))

        dataset_loader.preprocess_datasets();
        try {
            test_menu(model_handler, dataset_loader, eval_db);
        } catch (Exception e) {
            e.printStackTrace();
        }

        //run_all_tests(model_handler, dataset_loader);
    }

    public static void main(String[] args) {
        // Setup our objects
        EvaluationsDB eval_db = new EvaluationsDB();
        WekaPackageManager.loadPackages(false);
        ModelHandler model_handler = new ModelHandler();
        DatasetLoader dataset_loader = new DatasetLoader();

        try {
            eval_db.connect();
            eval_db.startup();
        } catch (SQLException e) {
            e.printStackTrace();
        }

        main_menu(model_handler, dataset_loader, eval_db);
    }
}
