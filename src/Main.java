import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.core.WekaPackageManager;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class Main {

    private static String[] coral_filepaths = {
            "datasets/promise/coral/cm1-jm1.arff",
            "datasets/promise/coral/cm1-kc1.arff",
            "datasets/promise/coral/cm1-kc2.arff",
            "datasets/promise/coral/cm1-pc1.arff",
            "datasets/promise/coral/jm1-kc1.arff",
            "datasets/promise/coral/jm1-kc2.arff",
            "datasets/promise/coral/jm1-pc1.arff",
            "datasets/promise/coral/kc1-jm1.arff",
            "datasets/promise/coral/kc1-kc2.arff",
            "datasets/promise/coral/kc1-pc1.arff",
            "datasets/promise/coral/kc2-cm1.arff",
            "datasets/promise/coral/kc2-jm1.arff",
            "datasets/promise/coral/kc2-pc1.arff",
            "datasets/promise/coral/pc1-cm1.arff",
            "datasets/promise/coral/pc1-jm1.arff",
            "datasets/promise/coral/pc1-kc1.arff"
    };

    private static String[] mmd_filepaths = {
            "datasets/promise/mmd/cm1-jm1.arff",
            "datasets/promise/mmd/cm1-kc1.arff",
            "datasets/promise/mmd/cm1-kc2.arff",
            "datasets/promise/mmd/cm1-pc1.arff",
            "datasets/promise/mmd/jm1-kc1.arff",
            "datasets/promise/mmd/jm1-kc2.arff",
            "datasets/promise/mmd/jm1-pc1.arff",
            "datasets/promise/mmd/kc1-jm1.arff",
            "datasets/promise/mmd/kc1-kc2.arff",
            "datasets/promise/mmd/kc1-pc1.arff",
            "datasets/promise/mmd/kc2-cm1.arff",
            "datasets/promise/mmd/kc2-jm1.arff",
            "datasets/promise/mmd/kc2-pc1.arff",
            "datasets/promise/mmd/pc1-cm1.arff",
            "datasets/promise/mmd/pc1-jm1.arff",
            "datasets/promise/mmd/pc1-kc1.arff"
    };

    // Simple test without feature selection
    public static List<EvaluationResult> run_tests (ModelHandler model_handler, DatasetLoader dataset_loader)
            throws Exception {
        TestRunner test_runner = new TestRunner();
        List<EvaluationResult> evaluations = test_runner.run_cpdp_test(model_handler, dataset_loader.get_datasets());
//            System.out.println("Test Evaluations");
//            System.out.println(test_runner.evaluation_results_to_string(evaluations));
        System.out.println("\nTest Summarisations");
        System.out.println(test_runner.summarise_results_per_training_set(evaluations) + "\n");
        System.out.println(test_runner.summarise_results(evaluations));
        return evaluations;
    }

    // Test with feature selection
    public static List<EvaluationResult> run_tests (
            ModelHandler model_handler, DatasetLoader dataset_loader,
            String evaluator, String search_method,
            Double start_threshold, Double end_threshold, Double step) throws Exception {
        TestRunner test_runner = new TestRunner();
        String fs_details = "Evaluator: " + evaluator + " Search Method: " + search_method;
        System.out.println("Beginning test for " + fs_details);

        List<EvaluationResult> evaluations = test_runner.run_cpdp_test(
                model_handler, dataset_loader.get_datasets(),
                evaluator, search_method,
                start_threshold, end_threshold, step
        );

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
                    List<EvaluationResult> evaluations;
                    if (search_method_entry.getKey().equals("Ranker")) {
                        evaluations = run_tests(model_handler, dataset_loader, evaluator_entry.getKey(), search_method_entry.getKey(), 0.02, 0.1, 0.01);
                    } else {
                        evaluations = run_tests(model_handler, dataset_loader, evaluator_entry.getKey(), search_method_entry.getKey(), null, null, null);
                    }

                    prompt_user_save_to_db(evaluations, eval_db);
                } catch (Exception e) {
                    System.out.println("Could not run test using " + fs_details);
                    e.printStackTrace();
                }
            }
        }
    }

    public static double[] input_threshold_values () {
        Scanner scanner = new Scanner(System.in);
        double[] thresholds = new double[3];
        System.out.println("Enter starting threshold value: ");
        thresholds[0] = scanner.nextDouble();
        System.out.println("Enter end threshold value: ");
        thresholds[1] = scanner.nextDouble();
        System.out.println("Enter threshold step value: ");
        thresholds[2] = scanner.nextDouble();
        return thresholds;
    }


    public static void aligned_test (String[] filepaths, String da_type, EvaluationsDB eval_db) throws Exception {
        DatasetLoader dataset_loader = new DatasetLoader();
        dataset_loader.load_aligned_datasets(filepaths);

        TestRunner runner = new TestRunner();
        ModelHandler model_handler = new ModelHandler();

        System.out.printf("%n%s aligned tests beginning!%n", da_type);

        // No feature selection
        List<EvaluationResult> results = runner.run_aligned_cpdp_test(model_handler, dataset_loader, da_type);
        System.out.println(runner.evaluation_results_to_string(results));
        System.out.println("\nTest Summarisations");
        System.out.println(runner.summarise_results_per_training_set(results) + "\n");
        System.out.println(runner.summarise_results(results));
        System.out.println("Saving results!");
        eval_db.insert_evaluations(results);

        // Feature selection
        FeatureSelection feature_selection = new FeatureSelection();
        for (Map.Entry<String, ASEvaluation> evaluator_entry : feature_selection.get_evaluators().entrySet()) {
            for (Map.Entry<String, ASSearch> search_method_entry : feature_selection.get_search_methods().entrySet()) {
                String fs_details = "Evaluator: " + evaluator_entry.getKey() + " Search Method: " + search_method_entry.getKey();
                List<EvaluationResult> evaluations = null;
                try {
                    if (search_method_entry.getKey().equals("Ranker")) {
                        evaluations = runner.run_aligned_cpdp_test(
                                model_handler, dataset_loader, da_type,
                                evaluator_entry.getKey(), search_method_entry.getKey(),
                                0.02, 0.1, 0.01
                        );
                    } else {
                        evaluations = runner.run_aligned_cpdp_test(
                                model_handler, dataset_loader, da_type,
                                evaluator_entry.getKey(), search_method_entry.getKey()
                        );
                    }

                    //prompt_user_save_to_db(evaluations, eval_db);
                } catch (Exception e) {
                    System.out.println("Could not run test using " + fs_details);
                    e.printStackTrace();
                }

                if(!evaluations.isEmpty()) {
                    System.out.println("Saving results!");
                    eval_db.insert_evaluations(evaluations);
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
        System.out.println("5: CORAL");
        System.out.println("6: MMD");

        Scanner scanner = new Scanner(System.in);
        int user_input = scanner.nextInt();
        List<EvaluationResult> evaluations = new ArrayList<>();
        double[] thresholds = new double[3];
        switch (user_input) {
            case 1: // No Feature Selection
                evaluations = run_tests(model_handler, dataset_loader);
                break;
            case 2: // CFS
                evaluations = run_tests(
                        model_handler, dataset_loader, "CFS Subset", "Best First",
                        1.0, 1.0, 1.0
                );
                break;
            case 3: // Info Gain
                thresholds = input_threshold_values();
                evaluations = run_tests(
                        model_handler, dataset_loader, "Info Gain", "Ranker",
                        thresholds[0], thresholds[1], thresholds[2]
                );
                break;
            case 4: // Gain Ratio
                thresholds = input_threshold_values();
                evaluations = run_tests(
                        model_handler, dataset_loader, "Gain Ratio", "Ranker",
                        thresholds[0], thresholds[1], thresholds[2]
                );
                break;

            case 5: // CORAL Tests
                aligned_test(coral_filepaths, "Coral", eval_db);
                break;
            case 6: // MMD Tests
                aligned_test(mmd_filepaths, "MMD", eval_db);
                break;
            default:
                break;
        }
        prompt_user_save_to_db(evaluations, eval_db);
    }

    public static void main_menu (ModelHandler model_handler, DatasetLoader dataset_loader, EvaluationsDB eval_db) throws Exception {
        Scanner scanner = new Scanner(System.in);

        // First we must check if tests should be done with preprocessing
        System.out.println("Preprocess missing values? (Y/N)");
        String user_input = scanner.nextLine().trim().toLowerCase();
        if (user_input.equals("y")) dataset_loader.preprocess_datasets();

        boolean menu_active = true;

        while (menu_active) {
            System.out.println("\n1: Test Menu");
            System.out.println("2: Wipe Database");
            System.out.println("3: Summarise Evaluations");
            System.out.println("4: Top Performing Models");
            System.out.println("5: Exit");
            int menu_input = scanner.nextInt();

            switch (menu_input) {
                case 1: // Test Menu
                    try {
                        test_menu(model_handler, dataset_loader, eval_db);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    break;
                case 2: // Wipe Database
                    try {
                        eval_db.delete_evaluations();
                        System.out.println("All records successfully deleted from database!");
                    } catch (SQLException e) {
                        System.out.println("Error deleting records from database!");
                        e.printStackTrace();
                    }
                    break;
                case 3: // Get summaries from database
                    try {
                        eval_db.print_evaluation_summary();
                    } catch (SQLException e) {
                        e.printStackTrace();
                    }
                    break;
                case 4:
                    System.out.println("Choose metric to rank models by");
                    System.out.println("1: Accuracy");
                    System.out.println("2: Recall");
                    System.out.println("3: F-Measure");
                    int metric_input = scanner.nextInt();
                    String metric = null;
                    switch (metric_input) {
                        case 1:
                            metric = "accuracy";
                            break;
                        case 2:
                            metric = "recall";
                            break;
                        case 3:
                            metric = "f_measure";
                            break;
                    }
                    try {
                        eval_db.print_top_performing_models(metric);
                    } catch (SQLException e) {
                        e.printStackTrace();
                    }
                    break;
                case 5: // Quit
                    menu_active = false;
                    break;
                default:
                    break;

            }
        }
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

//        try {
//            // First we must check if tests should be done with preprocessing
//            dataset_loader.preprocess_datasets();
//
//            aligned_test(coral_filepaths, "Coral", eval_db);
//            aligned_test(mmd_filepaths, "MMD", eval_db);
//        } catch (Exception e) {
//            e.printStackTrace();
//        }

        try {
            main_menu(model_handler, dataset_loader, eval_db);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
