import sys
import pandas as pd


def load_matrix(excel_path):
    """Load a row-activity x column-activity score matrix from an Excel file."""
    return pd.read_excel(excel_path, index_col=0)


def best_matches_per_column(matrix):
    """Given a matrix with scores, return column & row pairs where the row had the highest score for the column.
    :return: [ ( <column_id>, <row_id>, <score> ), ... ]
    """
    return [
        (col, row, score)
        for col in matrix.columns
        for row, score in matrix[col].items()
        if score == matrix[col].max()
    ]

def best_matches_per_column_row(matrix, threshold):
    """Given a matrix with scores, return column & row pairs where the row had the highest score for the column, 
    *and* the column had the highest score for that row.
    :return: [ ( <column_id>, <row_id>, <score> ), ... ]
    """
    results = []
    for col in matrix.columns:
        col_scores = matrix[col] # scores for node2 (col)
        best_score = col_scores.max() # best score for node2
        for row, score in col_scores.items(): # for node1 (row), score
            # if this is not best score for node2, continue
            if score != best_score:
                continue
            # if not best score for node1, continue
            other_best = matrix.loc[row].drop(col).max()
            if other_best > score:
                continue
            # else, if score is over threshold, 
            # record node1 <> node2 match
            if score >= threshold:
                results.append((col, row, score))
    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python {sys.argv[0]} <matrix.xlsx>")

    matrix = load_matrix(sys.argv[1])
    matches = best_matches_per_column_row(matrix)
    print(matches)
    for col, row, score in matches:
        print(f"({col}, {row}, {score})")
