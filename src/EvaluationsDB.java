import weka.classifiers.Evaluation;
import java.sql.*;
import java.util.List;
import java.util.Map;

public class EvaluationsDB {
    // Constants
    private static final String db_name = "evaluations.db";
    private static final String url = "jdbc:sqlite:" + db_name;
    private Connection conn;

    private static final String create_evaluation_table_query = """
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluator TEXT,
            search_method TEXT,
            da_type TEXT,
            threshold REAL,
            model_name TEXT NOT NULL,
            training_set TEXT NOT NULL,
            testing_set TEXT NOT NULL,
            accuracy REAL NOT NULL,
            recall REAL NOT NULL,
            f_measure REAL NOT NULL
        );
    """;

    private static final String insert_evaluation_query = """
        INSERT INTO evaluations (
            model_name, da_type, evaluator, search_method, threshold, training_set, testing_set, accuracy, recall, f_measure
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
    """;

    private static final String delete_evaluations_query = """
            DELETE FROM evaluations;
    """;

    private static final String get_evaluation_summary = """
        SELECT 
            da_type, evaluator, search_method, model_name, threshold, COUNT(id) as test_count, 
            AVG(accuracy) as avg_accuracy, AVG(recall) as avg_recall, AVG(f_measure) as avg_f_measure
        FROM evaluations
        GROUP BY da_type, evaluator, search_method, model_name, threshold;
    """;

    private static final String model_performance_ranking_query = """
        SELECT 
            model_name, da_type, evaluator, search_method, threshold, 
            AVG(accuracy) as accuracy, AVG(recall) as recall, AVG(f_measure) as f_measure
        FROM evaluations
        GROUP BY da_type, evaluator, search_method, model_name, threshold
        HAVING AVG(recall) >= 0.3
        ORDER BY %s DESC
        LIMIT 10;
    """;
    public void connect () throws SQLException {
        if (conn == null || conn.isClosed()) {
            conn = DriverManager.getConnection(url);
        }
    }

    public void disconnect() throws SQLException {
        if (conn != null && !conn.isClosed()) {
            conn.close();
        }
    }

    public void startup () throws SQLException {
        connect();
        var statement = conn.createStatement();
        statement.execute(create_evaluation_table_query);
        disconnect();
    }

    public void insert_evaluation (
            String model_name, String da_type,
            String evaluator, String search_method, Double threshold,
            String training_set_name, String testing_set_name,
            Double accuracy, Double recall, Double f_measure) throws SQLException {

        connect();
        PreparedStatement prep_statement = conn.prepareStatement(insert_evaluation_query);
        prep_statement.setString(1, model_name);
        prep_statement.setString(2, da_type);
        prep_statement.setString(3, evaluator);
        prep_statement.setString(4, search_method);
        prep_statement.setDouble(5, threshold);
        prep_statement.setString(6, training_set_name);
        prep_statement.setString(7, testing_set_name);
        prep_statement.setDouble(8, accuracy);
        prep_statement.setDouble(9, recall);
        prep_statement.setDouble(10, f_measure);
        prep_statement.executeUpdate();
        disconnect();
    }

    // Convenience method for easy insertion for map entries
    public void insert_evaluation (EvaluationResult evaluation) throws SQLException {
        insert_evaluation(
                evaluation.get_model_name(), evaluation.get_da_type(),
                evaluation.get_evaluator(), evaluation.get_search_method(), evaluation.get_threshold(),
                evaluation.get_training_set_name(), evaluation.get_testing_set_name(),
                evaluation.get_accuracy(), evaluation.get_recall(), evaluation.get_f_measure()
        );
    }

    // Allows insertion of a list of evaluations
    public void insert_evaluations (List<EvaluationResult> evaluations) throws SQLException {
        for (EvaluationResult evaluation : evaluations) {
            insert_evaluation(evaluation);
        }
    }

    public void delete_evaluations () throws SQLException {
        connect();
        var statement = conn.createStatement();
        statement.execute(delete_evaluations_query);
        disconnect();
    }

    public ResultSet get_evaluation_summary () throws SQLException {
        connect();
        var statement = conn.createStatement();
        ResultSet results = statement.executeQuery(get_evaluation_summary);
        disconnect();
        return results;
    }



    public ResultSet get_top_performing_models (String metric) throws SQLException {
        // Ensure the metric is one of the allowed columns to avoid SQL injection
        if (!metric.equals("accuracy") && !metric.equals("recall") && !metric.equals("f_measure")) {
            throw new IllegalArgumentException("Invalid metric: " + metric);
        } else {
            String query = String.format(model_performance_ranking_query, metric);

            connect();
            PreparedStatement prepStatement = conn.prepareStatement(query);
            ResultSet results = prepStatement.executeQuery();
            return results;
        }
    }

    public void print_top_performing_models (String metric) throws SQLException {
        ResultSet results = get_top_performing_models(metric);

        // Print the header
        System.out.printf(
                "%n%-20s %-20s %-20s %-20s %-10s %-10s %-10s %-10s %-10s%n",
                "Model Name", "DA Type", "Evaluator", "Search Method", "Threshold", "Tests", "Accuracy", "Recall", "F-Measure"
        );
        System.out.println("-----------------------------------------------------------------------------------------------------------------------------------------");

        while (results.next()) {
            System.out.printf("%-20s %-20s %-20s %-20s %-20.2f %-10.4f %-10.4f %-10.4f%n",
                    results.getString("model_name"),
                    results.getString("da_type"),
                    results.getString("evaluator"),
                    results.getString("search_method"),
                    results.getDouble("threshold"),
                    results.getDouble("accuracy"),
                    results.getDouble("recall"),
                    results.getDouble("f_measure")
            );
        }
        disconnect();
    }

    public void print_evaluation_summary() throws SQLException {
        connect();
        var statement = conn.createStatement();
        ResultSet results = statement.executeQuery(get_evaluation_summary);

        // Print the header
        System.out.printf(
                "%n%-20s %-20s %-20s %-20s %-10s %-10s %-10s %-10s %-10s%n",
                "DA Type", "Evaluator", "Search Method", "Model Name", "Threshold", "Tests", "Accuracy", "Recall", "F-Measure"
        );
        System.out.println("-----------------------------------------------------------------------------------------------------------------------------------------");

        // Iterate through the results
        while (results.next()) {
            System.out.printf(
                    "%-20s %-20s %-20s %-20s %-10.2f %-10d %-10.4f %-10.4f %-10.4f%n",
                    results.getString("da_type"),
                    results.getString("evaluator"),
                    results.getString("search_method"),
                    results.getString("model_name"),
                    results.getDouble("threshold"),
                    results.getInt("test_count"),
                    results.getDouble("avg_accuracy"),
                    results.getDouble("avg_recall"),
                    results.getDouble("avg_f_measure")
            );
        }
        disconnect();
    }
}
