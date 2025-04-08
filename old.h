#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;
using namespace py::literals;

static constexpr unsigned int RESERVED_INDICES[] = { 0 };
static const py::bool_ py_true(true);



py::tuple generate_training_set(const unsigned int max_input_size, const uint64_t dataset_size, const int max_lookahead, const unsigned int max_edges, const py::object& reserved_inputs, const int distance_from_start, const int max_prefix_vertices, const bool quiet=false)
{

    std::cout << "Generating training set with max_input_size=" << max_input_size
         << ", dataset_size=" << dataset_size
         << ", max_lookahead=" << max_lookahead
         << ", max_edges=" << max_edges
         << ", distance_from_start=" << distance_from_start
         << ", max_prefix_vertices=" << max_prefix_vertices
         << std::endl;

	const unsigned int QUERY_PREFIX_TOKEN = (max_input_size-5) / 3 + 4;
	const unsigned int PADDING_TOKEN = (max_input_size-5) / 3 + 3;
	const unsigned int EDGE_PREFIX_TOKEN = (max_input_size-5) / 3 + 2;
	const unsigned int PATH_PREFIX_TOKEN = (max_input_size-5) / 3 + 1;

	unsigned int num_generated = 0;
	unsigned int num_collisions = 0;
	unsigned int ntokens = (max_input_size - 5) / 3 + 5;
	size_t input_shape[2]{dataset_size, max_input_size};
	size_t output_shape[2]{dataset_size, ntokens};
	size_t label_shape[1]{dataset_size};
	py::array_t<int64_t, py::array::c_style> inputs(input_shape);
	py::array_t<float, py::array::c_style> outputs(output_shape);
	py::array_t<int64_t, py::array::c_style> labels(label_shape);
	auto inputs_mem = inputs.mutable_unchecked<2>();
	auto outputs_mem = outputs.mutable_unchecked<2>();
	auto labels_mem = labels.mutable_unchecked<1>();
	unsigned int* lookahead_step_histogram = (unsigned int*) alloca(sizeof(unsigned int) * max_input_size);
	unsigned int* path_length_histogram = (unsigned int*) alloca(sizeof(unsigned int) * max_input_size);
	for (unsigned int i = 0; i < max_input_size; i++) {
		lookahead_step_histogram[i] = 0;
		path_length_histogram[i] = 0;
	}
	float* MAX_FREQS_PER_BUCKET = (float*) alloca(sizeof(float) * max_input_size);
	if (max_lookahead == -1) {
		for (unsigned int i = 0; i < max_input_size; i++)
			MAX_FREQS_PER_BUCKET[i] = 1.0;
	} else {
		for (unsigned int i = 0; i < (unsigned) max_lookahead + 1; i++)
			MAX_FREQS_PER_BUCKET[i] = 1.0 / (max_lookahead+1);
		for (unsigned int i = max_lookahead + 1; i < max_input_size; i++)
			MAX_FREQS_PER_BUCKET[i] = 0.0;
		MAX_FREQS_PER_BUCKET[max_lookahead] += 0.05;
	}

	unsigned int* potential_lookaheads = (unsigned int*) alloca(max((size_t) 1, sizeof(unsigned int) * (max_lookahead + 1)));
	unsigned int potential_lookahead_count = 0;
	while (num_generated < dataset_size) {
		array<node> g(32);
		node* start; node* end;
		array<array<node*>> paths(8);
		while (true) {
			if (max_lookahead == -1) {
				unsigned int num_vertices = randrange(3, (max_input_size - 5) / 3);
				if (!generate_example(g, start, end, paths, num_vertices, 4, (max_input_size - 5) / 3, true, -1, 0, max_prefix_vertices == -1 ? max_input_size : max_prefix_vertices)) {
					for (node& n : g) core::free(n);
					for (array<node*>& a : paths) core::free(a);
					g.length = 0; paths.length = 0;
					continue;
				}
			} else {
				potential_lookahead_count = 0;
				for (unsigned int i = 0; i < (unsigned) max_lookahead + 1; i++)
					if (num_generated == 0 || lookahead_step_histogram[i] / num_generated < MAX_FREQS_PER_BUCKET[i])
						potential_lookaheads[potential_lookahead_count++] = i;
				unsigned int lookahead = choice(potential_lookaheads, potential_lookahead_count);

				unsigned int num_paths;
				if (lookahead == 0) {
					num_paths = randrange(1, 3);
				} else {
					unsigned int max_num_paths = (max_edges - 1) / lookahead;
					num_paths = randrange(2, max_num_paths + 1);
				}

				unsigned int num_vertices = std::min(std::min(lookahead * num_paths + 1 + randrange(0, 6), (max_input_size-5) / 3), max_edges + 1);
				if (!generate_example(g, start, end, paths, num_vertices, 4, (max_input_size - 5) / 3, true, lookahead, num_paths, max_prefix_vertices == -1 ? max_input_size : max_prefix_vertices)) {
					for (node& n : g) core::free(n);
					for (array<node*>& a : paths) core::free(a);
					g.length = 0; paths.length = 0;
					continue;
				}
			}
			unsigned int shortest_path_length = paths[0].length;
			for (unsigned int i = 1; i < paths.length; i++)
				if (paths[i].length < shortest_path_length)
					shortest_path_length = paths[i].length;
			if (shortest_path_length > 1)
				break;
			for (node& n : g) core::free(n);
			for (array<node*>& a : paths) core::free(a);
			g.length = 0; paths.length = 0;
		}

		array<pair<unsigned int, unsigned int>> edges(8);
		for (node& vertex : g)
			for (node* child : vertex.children)
				edges.add(make_pair(vertex.id, child->id));
		if (edges.length > max_edges || edges.length * 3 + 4 > max_input_size) {
			for (node& n : g) core::free(n);
			for (array<node*>& a : paths) core::free(a);
			g.length = 0; paths.length = 0;
			continue;
		}
		shuffle(edges);

		array<unsigned int> prefix(max_input_size);
		for (auto& entry : edges) {
			prefix[prefix.length++] = EDGE_PREFIX_TOKEN;
			prefix[prefix.length++] = entry.key;
			prefix[prefix.length++] = entry.value;
		}
		prefix[prefix.length++] = QUERY_PREFIX_TOKEN;
		prefix[prefix.length++] = start->id;
		prefix[prefix.length++] = end->id;
		prefix[prefix.length++] = PATH_PREFIX_TOKEN;

		for (const array<node*>& path : paths) {
			if (path.length == 1)
				continue;
			for (unsigned int j = 1; j < path.length; j++) {
				if (distance_from_start != -1 && j != (unsigned int) distance_from_start)
					continue;
				array<unsigned int> example(prefix.length + j);
				for (unsigned int i = 0; i < prefix.length; i++)
					example[i] = prefix[i];
				for (unsigned int i = 0; i < j; i++)
					example[prefix.length + i] = path[i]->id;
				example.length = prefix.length + j;
				if (example.length > max_input_size)
					continue;

				unsigned int lookahead_steps = lookahead_depth(path[j-1], path[j], end);
				array<node*> useful_steps(8);
				for (node* v : path[j-1]->children)
					if (has_path(v, end)) useful_steps.add(v);

				/* check if this input is reserved */
				py::object contains = reserved_inputs.attr("__contains__");
				py::tuple example_tuple(example.length);
				for (unsigned int i = 0; i < example.length; i++)
					example_tuple[i] = example[i];
				if (contains(example_tuple).is(py_true)) {
					num_collisions += 1;
					continue;
				}

				if (num_generated != 0 && lookahead_step_histogram[lookahead_steps] / num_generated >= MAX_FREQS_PER_BUCKET[lookahead_steps])
					continue;
				lookahead_step_histogram[lookahead_steps] += 1;
				path_length_histogram[j] += 1;

				for (unsigned int i = 0; i < max_input_size - example.length; i++)
					inputs_mem(num_generated, i) = PADDING_TOKEN;
				for (unsigned int i = 0; i < example.length; i++)
					inputs_mem(num_generated, max_input_size - example.length + i) = example[i];
				for (unsigned int i = 0; i < ntokens; i++)
					outputs_mem(num_generated, i) = 0.0f;
				for (unsigned int i = 0; i < useful_steps.length; i++)
					outputs_mem(num_generated, useful_steps[i]->id) = 1.0f;
				labels_mem(num_generated) = choice(useful_steps.data, useful_steps.length)->id;
				num_generated++;
				if (num_generated == dataset_size)
					break;
			}
			if (num_generated == dataset_size)
				break;
		}

		if (!quiet && num_generated > 0 && (num_generated % 1000 == 0 || num_generated >= dataset_size)) {
			printf("%d examples generated.\n", num_generated);

			printf("Lookahead steps histogram:\n");
			printf("[");
			bool first = true;
			for (unsigned int i = 0; i < max_input_size; i++) {
				if (lookahead_step_histogram[i] == 0)
					continue;
				if (!first) printf(", ");
				printf("%d:%.2f", i, (float) lookahead_step_histogram[i] / num_generated + 1e-9);
				first = false;
			}
			printf("]\n");

			printf("Path length histogram:\n");
			printf("[");
			first = true;
			for (unsigned int i = 0; i < max_input_size; i++) {
				if (path_length_histogram[i] == 0)
					continue;
				if (!first) printf(", ");
				printf("%d:%.2f", i, (float) path_length_histogram[i] / num_generated + 1e-9);
				first = false;
			}
			printf("]\n");
			fflush(stdout);
		}

		for (node& n : g) core::free(n);
		for (array<node*>& a : paths) core::free(a);
		g.length = 0; paths.length = 0;
		continue;
	}

	return py::make_tuple(inputs, outputs, labels, num_collisions);
}

py::array_t<int64_t, py::array::c_style> lookahead_histogram(const unsigned int max_input_size, const uint64_t num_samples, const unsigned int max_edges, const int distance_from_start, const bool quiet=false)
{
	const unsigned int QUERY_PREFIX_TOKEN = (max_input_size-5) / 3 + 4;
	const unsigned int EDGE_PREFIX_TOKEN = (max_input_size-5) / 3 + 2;
	const unsigned int PATH_PREFIX_TOKEN = (max_input_size-5) / 3 + 1;

	unsigned int num_generated = 0;
	unsigned int max_lookahead = ((max_input_size - 5) / 3 - 1) / 2;
	size_t histogram_shape[1]{max_lookahead};
	py::array_t<int64_t, py::array::c_style> histogram(histogram_shape);
	auto histogram_mem = histogram.mutable_unchecked<1>();
	for (unsigned int i = 0; i < max_lookahead; i++)
		histogram_mem(i) = 0;

	while (num_generated < num_samples) {
		array<node> g(32);
		node* start; node* end;
		array<array<node*>> paths(8);
		while (true) {
			unsigned int num_vertices = randrange(3, (max_input_size - 5) / 3);
			if (!generate_example(g, start, end, paths, num_vertices, 4, (max_input_size - 5) / 3, true, -1, 0, -1)) {
				for (node& n : g) core::free(n);
				for (array<node*>& a : paths) core::free(a);
				g.length = 0; paths.length = 0;
				continue;
			}
			unsigned int shortest_path_length = paths[0].length;
			for (unsigned int i = 1; i < paths.length; i++)
				if (paths[i].length < shortest_path_length)
					shortest_path_length = paths[i].length;
			if (shortest_path_length > 1)
				break;
			for (node& n : g) core::free(n);
			for (array<node*>& a : paths) core::free(a);
			g.length = 0; paths.length = 0;
		}

		array<pair<unsigned int, unsigned int>> edges(8);
		for (node& vertex : g)
			for (node* child : vertex.children)
				edges.add(make_pair(vertex.id, child->id));
		if (edges.length > max_edges || edges.length * 3 + 4 > max_input_size) {
			for (node& n : g) core::free(n);
			for (array<node*>& a : paths) core::free(a);
			g.length = 0; paths.length = 0;
			continue;
		}
		shuffle(edges);

		array<unsigned int> prefix(max_input_size);
		for (auto& entry : edges) {
			prefix[prefix.length++] = EDGE_PREFIX_TOKEN;
			prefix[prefix.length++] = entry.key;
			prefix[prefix.length++] = entry.value;
		}
		prefix[prefix.length++] = QUERY_PREFIX_TOKEN;
		prefix[prefix.length++] = start->id;
		prefix[prefix.length++] = end->id;
		prefix[prefix.length++] = PATH_PREFIX_TOKEN;

		for (const array<node*>& path : paths) {
			if (path.length == 1)
				continue;
			for (unsigned int j = 1; j < path.length; j++) {
				if (distance_from_start != -1 && j != (unsigned int) distance_from_start)
					continue;
				array<unsigned int> example(prefix.length + j);
				for (unsigned int i = 0; i < prefix.length; i++)
					example[i] = prefix[i];
				for (unsigned int i = 0; i < j; i++)
					example[prefix.length + i] = path[i]->id;
				example.length = prefix.length + j;
				if (example.length > max_input_size)
					continue;

				unsigned int lookahead_steps = lookahead_depth(path[j-1], path[j], end);
				histogram_mem(lookahead_steps) += 1;
				num_generated++;
				if (num_generated == num_samples)
					break;
			}
			if (num_generated == num_samples)
				break;
		}

		if (!quiet && num_generated > 0 && (num_generated % 1000 == 0 || num_generated >= num_samples)) {
			printf("%d examples generated.\n", num_generated);

			printf("Lookahead steps histogram:\n");
			printf("[");
			bool first = true;
			for (unsigned int i = 0; i < max_lookahead; i++) {
				if (histogram_mem(i) == 0)
					continue;
				if (!first) printf(", ");
				printf("%d:%.2f", i, (float) histogram_mem(i) / num_generated + 1e-9);
				first = false;
			}
			printf("]\n");
		}

		for (node& n : g) core::free(n);
		for (array<node*>& a : paths) core::free(a);
		g.length = 0; paths.length = 0;
		continue;
	}

	return histogram;
}

py::tuple generate_reachable_training_set(const unsigned int max_input_size, const uint64_t dataset_size, const unsigned int lookahead, const unsigned int max_edges, const py::object& reserved_inputs, const int distance_from_start, const int reachable_distance, const unsigned int start_vertex_index, const bool exclude_start_vertex)
{
	const unsigned int QUERY_PREFIX_TOKEN = (max_input_size-5) / 3 + 4;
	const unsigned int PADDING_TOKEN = (max_input_size-5) / 3 + 3;
	const unsigned int EDGE_PREFIX_TOKEN = (max_input_size-5) / 3 + 2;
	const unsigned int PATH_PREFIX_TOKEN = (max_input_size-5) / 3 + 1;

	unsigned int num_generated = 0;
	unsigned int num_collisions = 0;
	size_t input_shape[2]{dataset_size, max_input_size};
	size_t output_shape[2]{dataset_size, max_input_size};
	py::array_t<int64_t, py::array::c_style> inputs(input_shape);
	py::array_t<float, py::array::c_style> outputs(output_shape);
	auto inputs_mem = inputs.mutable_unchecked<2>();
	auto outputs_mem = outputs.mutable_unchecked<2>();
	py::list valid_outputs;

	unsigned int max_vertex_id = (max_input_size - 5) / 3;
	while (num_generated < dataset_size) {
		array<node> g(32);
		node* start; node* end;
		array<array<node*>> paths(8);
		while (true) {
			unsigned int num_paths;
			if (lookahead == 0) {
				num_paths = randrange(1, 3);
			} else {
				unsigned int max_num_paths = (max_edges - 1) / lookahead;
				num_paths = randrange(2, max_num_paths + 1);
			}

			unsigned int num_vertices = std::min(std::min(lookahead * num_paths + 1 + randrange(0, 6), (max_input_size-5) / 3), max_edges + 1);
			if (!generate_example(g, start, end, paths, num_vertices, 4, max_vertex_id, true, lookahead, num_paths, -1)) {
				for (node& n : g) core::free(n);
				for (array<node*>& a : paths) core::free(a);
				g.length = 0; paths.length = 0;
				continue;
			}
			unsigned int shortest_path_length = paths[0].length;
			for (unsigned int i = 1; i < paths.length; i++)
				if (paths[i].length < shortest_path_length)
					shortest_path_length = paths[i].length;
			if (shortest_path_length > 1)
				break;
			for (node& n : g) core::free(n);
			for (array<node*>& a : paths) core::free(a);
			g.length = 0; paths.length = 0;
		}

		array<pair<unsigned int, unsigned int>> edges(8);
		for (node& vertex : g)
			for (node* child : vertex.children)
				edges.add(make_pair(vertex.id, child->id));
		if (edges.length > max_edges || edges.length * 3 + 4 > max_input_size) {
			for (node& n : g) core::free(n);
			for (array<node*>& a : paths) core::free(a);
			g.length = 0; paths.length = 0;
			continue;
		}
		shuffle(edges);

		array<unsigned int> prefix(max_input_size);
		for (auto& entry : edges) {
			prefix[prefix.length++] = EDGE_PREFIX_TOKEN;
			prefix[prefix.length++] = entry.key;
			prefix[prefix.length++] = entry.value;
		}
		prefix[prefix.length++] = QUERY_PREFIX_TOKEN;
		prefix[prefix.length++] = start->id;
		prefix[prefix.length++] = end->id;
		prefix[prefix.length++] = PATH_PREFIX_TOKEN;

		for (const array<node*>& path : paths) {
			if (path.length == 1)
				continue;
			for (unsigned int j = 1; j < path.length; j++) {
				if (distance_from_start != -1 && j != (unsigned int) distance_from_start)
					continue;
				array<unsigned int> example(prefix.length + j);
				for (unsigned int i = 0; i < prefix.length; i++)
					example[i] = prefix[i];
				for (unsigned int i = 0; i < j; i++)
					example[prefix.length + i] = path[i]->id;
				example.length = prefix.length + j;
				if (example.length > max_input_size)
					continue;

				/* compute the set of reachable vertices */
				node** vertex_id_map = (node**) calloc(max_vertex_id + 1, sizeof(node*));
				for (unsigned int i = 0; i < g.length; i++)
					vertex_id_map[g[i].id] = &g[i];
				array<unsigned int> reachable(16);
				array<pair<unsigned int, unsigned int>> stack(16);
				unsigned int start_vertex;
				if (example.length < start_vertex_index)
					start_vertex = start->id;
				else start_vertex = example[example.length - start_vertex_index];
				stack.add(make_pair(start_vertex, 0u));
				while (stack.length != 0) {
					pair<unsigned int, unsigned int> entry = stack.pop();
					unsigned int current_vertex = entry.key;
					unsigned int current_distance = entry.value;
					if (!reachable.contains(current_vertex))
						reachable.add(current_vertex);
					if (reachable_distance > 0 && current_distance + 1 <= (unsigned int) reachable_distance) {
						for (node* child : vertex_id_map[current_vertex]->children)
							stack.add(make_pair(child->id, current_distance + 1));
					} else if (reachable_distance < 0 && current_distance + 1 <= (unsigned int) -reachable_distance) {
						for (node* parent : vertex_id_map[current_vertex]->parents)
							stack.add(make_pair(parent->id, current_distance + 1));
					}
				}
				if (exclude_start_vertex)
					reachable.remove(reachable.index_of(start_vertex));

				array<node*> useful_steps(8);
				for (node* v : path[j-1]->children)
					if (has_path(v, end)) useful_steps.add(v);

				/* check if this input is reserved */
				py::object contains = reserved_inputs.attr("__contains__");
				py::tuple example_tuple(example.length);
				for (unsigned int i = 0; i < example.length; i++)
					example_tuple[i] = example[i];
				if (contains(example_tuple).is(py_true)) {
					num_collisions += 1;
					continue;
				}

				for (unsigned int i = 0; i < max_input_size - example.length; i++)
					inputs_mem(num_generated, i) = PADDING_TOKEN;
				for (unsigned int i = 0; i < example.length; i++)
					inputs_mem(num_generated, max_input_size - example.length + i) = example[i];
				for (unsigned int i = 0; i < max_input_size - example.length; i++)
					outputs_mem(num_generated, i) = 0;
				for (unsigned int i = 0; i < example.length; i++)
					outputs_mem(num_generated, max_input_size - example.length + i) = reachable.contains(example[i]) ? 1 : 0;
				num_generated++;
				if (num_generated == dataset_size)
					break;
			}
			if (num_generated == dataset_size)
				break;
		}

		for (node& n : g) core::free(n);
		for (array<node*>& a : paths) core::free(a);
		g.length = 0; paths.length = 0;
		continue;
	}

	return py::make_tuple(inputs, outputs, num_collisions);
}



PYBIND11_MODULE(generator, m) {
	m.def("generate_training_set", &generate_training_set);
	m.def("generate_reachable_training_set", &generate_reachable_training_set);
	m.def("generate_dfs_training_set", &generate_dfs_training_set);
	m.def("generate_si_training_set", &generate_si_training_set);
	m.def("lookahead_histogram", &lookahead_histogram);
	m.def("set_seed", &core::set_seed);
}
