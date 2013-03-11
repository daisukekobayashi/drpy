#define NDEBUG

#include <vector>
#include <limits>
#include <utility>
#include <pyublas/numpy.hpp>

#include <boost/multi_array.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/johnson_all_pairs_shortest.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>


pyublas::numpy_matrix<double>
geodesic_distance(pyublas::numpy_matrix<double> src, int k, double inf)
{
    using namespace boost;
    namespace ublas = boost::numeric::ublas;

    std::vector< std::pair<std::size_t, std::size_t> > edge;
    std::vector<double> weight;

    for (std::size_t i = 0; i < src.size1(); ++i) {

        src(i, i) = std::numeric_limits<double>::max();

        std::pair<std::size_t, std::size_t> tmp;
        tmp.first = i;

        ublas::matrix_row< pyublas::numpy_matrix<double> >::iterator iter;
        for (std::size_t j = 0; j < k; ++j) {
            iter = std::min_element(ublas::row(src, i).begin(), ublas::row(src, i).end());
            tmp.second = iter.index();
            edge.push_back(tmp);
            weight.push_back(src(tmp.first, tmp.second));
            src(i, tmp.second) = std::numeric_limits<double>::max();
        }
    }

    typedef adjacency_list<vecS, vecS, undirectedS, no_property,
    property<edge_weight_t, double, property<edge_weight2_t, double > > > Graph;

    typedef std::pair<std::size_t, std::size_t> Edge;
    Edge* edge_array = new Edge[edge.size()];
    for (std::size_t i = 0; i < edge.size(); ++i) {
        edge_array[i] = edge[i];
    }

    const std::size_t num_vertices = src.size1();
    Graph graph(edge_array, edge_array + edge.size(), num_vertices);

    property_map<Graph, edge_weight_t>::type w = get(edge_weight, graph);
    graph_traits<Graph>::edge_iterator eit, eit_end;
    std::vector<double>::const_iterator cwit = weight.begin();
    for (tie(eit, eit_end) = edges(graph); eit != eit_end; ++eit) {
        w[*eit] = *cwit++;
    }

    std::vector<double> d(num_vertices, inf);
    multi_array<double, 2> dist(extents[num_vertices][num_vertices]);
    //double** dist = new double*[num_vertices];
    //for (std::size_t i = 0; i < num_vertices; i++) {
    //  dist[i] = new double[num_vertices];
    //}

    johnson_all_pairs_shortest_paths(graph, dist, distance_map(&d[0]));

    pyublas::numpy_matrix<double> gdist(num_vertices, num_vertices);
    for (std::size_t i = 0; i < gdist.size1(); ++i) {
        for (std::size_t j = 0; j < gdist.size2(); ++j) {
            gdist(i, j) = dist[i][j];
        }
    }

    //for (std::size_t i = 0; i < num_vertices; i++) {
    //  delete[] dist[i];
    //}
    //delete[] dist;
    delete[] edge_array;

    return gdist;
}


BOOST_PYTHON_MODULE(graph)
{
    boost::python::def("geodesic_distance", geodesic_distance);
}

