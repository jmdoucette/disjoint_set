use disjoint_set_forest::DisjointSet;

#[derive(Debug, PartialEq, Eq)]
struct Edge {
    u: usize,
    v: usize,
    weight: i32,
}

impl Edge {
    pub fn new(u: usize, v: usize, weight: i32) -> Self {
        Self {u, v, weight}
    }
}

fn kruskal(n: usize, mut edges: Vec<Edge>) -> Vec<Edge> {
    let mut msf = Vec::new();
    let mut ds = DisjointSet::new();
    for node in 0..n {
        ds.insert(node);
    }
    edges.sort_by_key(|edge| edge.weight);
    for edge in edges {
        if !ds.same_set(&edge.u, &edge.v).unwrap() {
            ds.union(&edge.u, &edge.v).unwrap();
            msf.push(edge);
        }
    }
    msf
}


#[test]
fn test_kruskal() {
    let edges1 = vec![
        Edge::new(0, 1, 7),
        Edge::new(0, 3, 5),
        Edge::new(1, 2, 8),
        Edge::new(1, 3, 9),
        Edge::new(1, 4, 7),
        Edge::new(2, 4, 5),
        Edge::new(3, 4, 15),
        Edge::new(3, 5, 6),
        Edge::new(4, 5, 8),
        Edge::new(4, 6, 9),
        Edge::new(5, 6, 11),
    ];
    let mut msf1 = kruskal(7, edges1);
    msf1.sort_by_key(|edge| (edge.u, edge.v));
    let expected1 = vec![
        Edge::new(0, 1, 7),
        Edge::new(0, 3, 5),
        Edge::new(1, 4, 7),
        Edge::new(2, 4, 5),
        Edge::new(3, 5, 6),
        Edge::new(4, 6, 9),
    ];
    assert_eq!(msf1, expected1);

    let edges2 = vec![
        Edge::new(0, 1, -2),
        Edge::new(0, 1, 3),
        Edge::new(0, 2, 8),
        Edge::new(0, 2, 100),
        Edge::new(1, 2, 7),
        Edge::new(1, 2, 7),
    ];
    let mut msf2 = kruskal(3, edges2);
    msf2.sort_by_key(|edge| (edge.u, edge.v));
    let expected2 = vec![
        Edge::new(0, 1, -2),
        Edge::new(1, 2, 7),
    ];
    assert_eq!(msf2, expected2);

}