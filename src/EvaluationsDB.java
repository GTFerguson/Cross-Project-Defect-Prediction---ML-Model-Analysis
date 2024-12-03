import weka.classifiers.Evaluation;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
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
            model_name, evaluator, search_method, threshold, training_set, testing_set, accuracy, recall, f_measure
        ) VALUES (?,?,?,?,?,?,?,?,?)
    """;

    public void connect () throws SQLException {
        if (conn == null || conn.isClosed()) {
            conn = DriverManager.getConnection(url);
            System.out.println("Connected to database: " + db_name);
        }
    }

    public void disconnect() throws SQLException {
        if (conn != null && !conn.isClosed()) {
            conn.close();
            System.out.println("Disconnected from database.");
        }
    }

    public void startup () throws SQLException {
        connect();
        var statement = conn.createStatement();
        statement.execute(create_evaluation_table_query);
        disconnect();
    }

    public void insert_evaluation (
            String model_name, String evaluator, String search_method, Double threshold,
            String training_set_name, String testing_set_name,
            Double accuracy, Double recall, Double f_measure) throws SQLException {
        connect();
        PreparedStatement prep_statement = conn.prepareStatement(insert_evaluation_query);
        prep_statement.setString(1, model_name);
        prep_statement.setString(2, evaluator);
        prep_statement.setString(3, search_method);
        prep_statement.setDouble(4, threshold);
        prep_statement.setString(5, training_set_name);
        prep_statement.setString(6, testing_set_name);
        prep_statement.setDouble(7, accuracy);
        prep_statement.setDouble(8, recall);
        prep_statement.setDouble(9, f_measure);
        prep_statement.executeUpdate();
        disconnect();
    }

    // Convenience method for easy insertion for map entries
    public void insert_evaluation (EvaluationResult evaluation) throws SQLException {
        Evaluation eval = evaluation.get_evaluation();
        System.out.println("Training Set: " + evaluation.get_training_set_name());
        insert_evaluation(
                evaluation.get_model_name(), evaluation.get_evaluator(),
                evaluation.get_search_method(), evaluation.get_threshold(),
                evaluation.get_training_set_name(), evaluation.get_testing_set_name(),
            eval.pctCorrect()/100, eval.recall(1), eval.fMeasure(1)
        );
    }

    public void insert_evaluations (List<EvaluationResult> evaluations) throws SQLException {
        for (EvaluationResult evaluation : evaluations) {
            insert_evaluation(evaluation);
        }
    }
}
