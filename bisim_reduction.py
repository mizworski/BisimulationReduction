import argparse
import re

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import lit

Node = Row("n_id", "p_id_0", "p_id_new")
Edge = Row("s_id", "t_id", "e_label", "p_id_old")


def read_input(input_path, spark):
  with open(input_path, 'r') as file:
    lines = file.readlines()
    m = re.match('^des \([ \t]*(\d+)[ \t]*,[ \t]*(\d+)[ \t]*,[ \t]*(\d+)[ \t]*\)', lines[0])
    initial_state, edges_num, nodes_num = map(int, m.groups())

    assert initial_state == 0

    edges = []

    nodes = [Node(node, 0, 0) for node in range(nodes_num)]

    for line in lines[1: edges_num + 1]:
      t_m = re.match('^[ \t]*\([ \t]*(\d+)[ \t]*,'
                     '[ \t]*([a-zA-Z][a-zA-Z0-9_]*|\"[ !\x23-\x7E]+\")[ \t]*,'
                     '[ \t]*(\d+)[ \t]*\)', line)
      source, e_label, target = int(t_m.group(1)), t_m.group(2), int(t_m.group(3))
      if e_label.startswith("\""):
        e_label = e_label[1:-1]

      edges.append(Edge(source, target, e_label, 0))

    nodes_df = spark.createDataFrame(nodes)
    edges_df = spark.createDataFrame(edges)

    nodes_df = nodes_df.alias('nodes')
    edges_df = edges_df.alias('edges')

    return nodes_df, edges_df


def task_signature(nodes, edges):
  def create_signature(values):
    # Check if values represent node.
    if values[0] == '0':
      return values[1], frozenset([])
    else:
      return None, frozenset({(values[0], values[2])})

  def merge_val(sig, values):
    if values[0] == '0':
      return values[1], sig[1]
    else:
      return sig[0], frozenset(sig[1].union({(values[0], values[2])}))

  def merge_combiners(sig1, sig2):
    merged_edges = frozenset(sig1[1].union(sig2[1]))
    if sig1[0] is not None:
      return sig1[0], merged_edges
    else:
      return sig2[0], merged_edges

  nodes_records = nodes.withColumn('e_label', lit(0)).select(['n_id', 'e_label', 'p_id_0', 'p_id_new'])

  t1 = edges.join(nodes, edges.t_id == nodes.n_id)
  t1_columns = ['s_id', 'e_label', 't_id', 'p_id_new']
  t1 = t1.select(t1_columns).withColumnRenamed("p_id_new", "p_id_old")

  t2 = t1.union(nodes_records).rdd
  t3 = t2.map(lambda r: (r[0], r[1:])).combineByKey(create_signature, merge_val, merge_combiners)

  return t3


def task_identifier(table):
  def identifier_mapper(rows):
    n_id, (p_id, sign) = rows
    return sign, (n_id, p_id)

  def create_list(nid_pid):
    n_id, p_id = nid_pid
    return n_id, [nid_pid]

  def merge_val(lst, val):
    n_id, p_id = val
    return min(lst[0], n_id), lst[1] + [val]

  def merge_lsts(lst1, lst2):
    # p_id_new here will be set to min(n_id) for n_ids with same signature.
    return min(lst1[0], lst2[0]), lst1[1] + lst2[1]

  def flatten_identifiers(val):
    _, (class_id, nodes) = val

    return [(n_id, (p_id_0, class_id)) for (n_id, p_id_0) in nodes]

  return table \
    .map(identifier_mapper) \
    .combineByKey(create_list, merge_val, merge_lsts) \
    .flatMap(flatten_identifiers)


def task_repartition(table, spark):
  sorted_table = table.sortByKey().map(lambda x: (x[0], x[1][0], x[1][1]))
  return spark.createDataFrame(sorted_table, ['n_id', 'p_id_0', 'p_id_new'])


def calculate_p_id_k(nodes, edges, k, spark, verbose=0):
  t1 = task_signature(nodes, edges)

  if verbose == 2:
    print("ITERATION {}".format(k))
    print("After signature task.")
    for row in t1.take(100):
      print(row)

  t2 = task_identifier(t1)

  if verbose == 2:
    print("After identifier task.")
    for row in t2.take(100):
      print(row)

  nodes_new = task_repartition(t2, spark)

  if verbose == 2:
    print("After repartiton task:")
    for row in nodes_new.rdd.take(100):
      print(row)

  if verbose:
    nodes_new.show()

  return nodes_new


def save_results(nodes, edges, output_file_path):
  edges_reduced = edges \
    .join(nodes, edges.s_id == nodes.n_id) \
    .withColumnRenamed('p_id_new', 's_id_new') \
    .select(['s_id_new', 'e_label', 't_id']) \
    .join(nodes, edges.t_id == nodes.n_id) \
    .withColumnRenamed('p_id_new', 't_id_new') \
    .select(['s_id_new', 'e_label', 't_id_new'])

  nodes = nodes.toPandas()
  assigned_p_ids = set(nodes['p_id_new'])
  new_p_ids = {p_id: i + 1 for i, p_id in enumerate(assigned_p_ids) if p_id != 0}
  new_p_ids[0] = 0

  edges = edges_reduced.toPandas().drop_duplicates().sort_values(['s_id_new', 't_id_new', 'e_label'])

  initial_state, edges_num, nodes_num = 0, len(edges), len(assigned_p_ids)

  with open(output_file_path, 'w+') as out_file:
    out_file.write("des ({}, {}, {})\n".format(initial_state, edges_num, nodes_num))

    for _, edge in edges.iterrows():
      out_file.write("({}, {}, {})\n".format(edge['s_id_new'], edge['e_label'], edge['t_id_new']))


def main():
  parser = argparse.ArgumentParser(description="Bisimulation Reduction of Big Graphs on MapReduce.")
  parser.add_argument("-i", "--input_path", default="data/ex1.aut", type=str,
                      help="Path to input file with graph in AUT format.")
  parser.add_argument("-o", "--output_path", default="data/ex1_reduced.aut", type=str,
                      help="Path to output file where reduced graph will be saved in AUT format.")
  parser.add_argument("-k", "--k_bisimiar", default=8, type=int,
                      help="Partition id will be calculated up to k bisimilarity.")
  parser.add_argument("-v", "--verbose", default=0, type=int,
                      help="Set to 0 for no logging, 1 for basic logging (N_t table only), 2 for full logging.")
  args = parser.parse_args()


  spark = SparkSession \
    .builder \
    .appName("Bisimulation Reduction of Big Graphs on MapReduce.") \
    .getOrCreate()

  # Notice: I don't allow using double quote character in label name (even though it's allowed in AUT format).
  # Other characters in ASCII range [32, 126] are allowed in quoted <quoted-label>.
  nodes, edges = read_input(args.input_path, spark)

  if args.verbose:
    nodes.show()

  if args.verbose == 2:
    print("Nodes as input.")
    for row in nodes.rdd.take(100):
      print(row)

  # In case k_bisimiar was set to 0
  nodes_new = nodes

  for k in range(args.k_bisimiar):
    nodes_new = calculate_p_id_k(nodes, edges, k, spark, args.verbose)

    # Check if p_id_k function has changed.
    # In 'nodes' table only p_id_new column is being changed (it represents p_id_k after k iterations).
    if nodes.subtract(nodes_new).rdd.isEmpty():
      break
    else:
      nodes = nodes_new

  save_results(nodes_new, edges, args.output_path)
  spark.stop()


if __name__ == '__main__':
  main()
