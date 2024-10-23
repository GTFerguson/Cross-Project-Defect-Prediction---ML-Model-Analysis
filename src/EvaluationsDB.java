import weka.classifiers.Evaluation;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.Map;

public class EvaluationsDB {
    private static final String db_name = "evaluations.db";
    private static final String url = "jdbc:sqlite:" + db_name;

    private static final String create_evaluation_table_query = """
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            training_set TEXT NOT NULL,
            testing_set TEXT NOT NULL,
            accuracy REAL NOT NULL,
            recall REAL NOT NULL,
            f_measure REAL NOT NULL
        );
    """;

    private static final String insert_evaluation_query = """
        INSERT INSERT INTO evaluations (
            model_name, training_set, testing_set, accuracy, recall, f_measure
        ) VALUES (?,?,?,?,?,?)
    """;


    public Connection connect () {
        Connection conn = null;
        try {
            conn = DriverManager.getConnection(url);
            if (conn != null) {
                System.out.println("Database has been created: " + db_name);
            }
        } catch (SQLException e) {
            System.err.println(e.getMessage());
        }
        return conn;
    }

    public void startup () throws SQLException {
        Connection conn = connect();

        if (conn != null) {
            var statement = conn.createStatement();
            statement.execute(create_evaluation_table_query);
        }

        conn.close();
    }

    public void insert_evaluation (
            String model_name, String training_set_name, String testing_set_name,
            Double accuracy, Double recall, Double f_measure
    ) throws SQLException {
        Connection conn = connect();

        if (conn != null) {
            PreparedStatement prep_statement = conn.prepareStatement(insert_evaluation_query);
            prep_statement.setString(1, model_name);
            prep_statement.setString(2, training_set_name);
            prep_statement.setString(3, testing_set_name);
            prep_statement.setDouble(4, accuracy);
            prep_statement.setDouble(5, recall);
            prep_statement.setDouble(6, f_measure);
            prep_statement.executeUpdate();
        }
    }

    // Override method for easy insertion for map entries
    public void insert_evaluation (Map.Entry<String, EvaluationResult> eval_result_entry) throws SQLException {
        EvaluationResult eval_result = eval_result_entry.getValue();
        Evaluation eval = eval_result.get_evaluation();

        insert_evaluation(eval_result_entry.getKey(), eval_result.get_training_set_name(), eval_result.get_training_set_name(),
                eval.pctCorrect()/100, eval.recall(1), eval.fMeasure(1)
        );
    }
}
