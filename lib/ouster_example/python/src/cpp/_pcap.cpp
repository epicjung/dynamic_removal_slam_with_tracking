/**
 * @file
 * @brief ouster_pyclient_pcap python module
 */
#include <pybind11/chrono.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <csignal>
#include <cstdlib>
#include <string>
#include <thread>

#include "ouster/os_pcap.h"
// Disabled until buffers are replaced
//#include "ouster/ouster_pybuffer.h"
#include <pcap/pcap.h>

#include <sstream>
PYBIND11_MAKE_OPAQUE(std::shared_ptr<ouster::sensor_utils::playback_handle>);
PYBIND11_MAKE_OPAQUE(std::shared_ptr<ouster::sensor_utils::record_handle>);

namespace py = pybind11;

/// @TODO The pybind11 buffer_info struct is actually pretty heavy-weight – it
/// includes some string fields, vectors, etc. Might be worth it to avoid
/// calling this twice.
/*
 * Check that buffer is a 1-d byte array of size > bound and return an internal
 * pointer to the data for writing. Check is strictly greater to account for the
 * extra byte required to determine if a datagram is bigger than expected.
 */
inline uint8_t* getptr(py::buffer& buf) {
    auto info = buf.request();
    if (info.format != py::format_descriptor<uint8_t>::format()) {
        throw std::invalid_argument(
            "Incompatible argument: expected a bytearray");
    }
    return (uint8_t*)info.ptr;
}
/// @TODO The pybind11 buffer_info struct is actually pretty heavy-weight – it
/// includes some string fields, vectors, etc. Might be worth it to avoid
/// calling this twice.
/*
 * Return the size of the python buffer
 */
inline size_t getptrsize(py::buffer& buf) {
    auto info = buf.request();

    return (size_t)info.size;
}

// Record functionality removed for a short amount of time
// until we switch it over to support libtins

PYBIND11_PLUGIN(_pcap) {
    py::module m("_pcap", R"(Pcap bindings generated by pybind11.

This module is generated directly from the C++ code and not meant to be used
directly.
)");

    // turn off signatures in docstrings: mypy stubs provide better types
    py::options options;
    options.disable_function_signatures();

    // clang-format off
    py::class_<ouster::sensor_utils::packet_info,
               std::shared_ptr<ouster::sensor_utils::packet_info>>(m, "packet_info")
        .def(py::init<>())
        .def("__repr__", [](const ouster::sensor_utils::packet_info& data) {
                             std::stringstream result;
                             result << data;
                             return result.str();})
        .def_readonly("dst_ip", &ouster::sensor_utils::packet_info::dst_ip)
        .def_readonly("src_ip", &ouster::sensor_utils::packet_info::src_ip)
        .def_readonly("dst_port", &ouster::sensor_utils::packet_info::dst_port)
        .def_readonly("src_port", &ouster::sensor_utils::packet_info::src_port)
        .def_property_readonly("timestamp",
             [](ouster::sensor_utils::packet_info& packet_info) -> double {
                 return packet_info.timestamp.count() / 1e6;
             })
        .def_readonly("payload_size", &ouster::sensor_utils::packet_info::payload_size)
        .def_readonly("fragments_in_packet",
                      &ouster::sensor_utils::packet_info::fragments_in_packet)
        .def_readonly("ip_version", &ouster::sensor_utils::packet_info::ip_version)
        .def_readonly("encapsulation_protocol", &ouster::sensor_utils::packet_info::encapsulation_protocol)
        .def_readonly("network_protocol", &ouster::sensor_utils::packet_info::network_protocol);
    
    
     py::class_<std::shared_ptr<ouster::sensor_utils::playback_handle>>(m, "playback_handle")
         .def("__init__", [](std::shared_ptr<ouster::sensor_utils::playback_handle>& self) {
             new (&self) std::shared_ptr<ouster::sensor_utils::playback_handle>{
                 ouster::sensor_utils::playback_handle_init()};
         });
    
    py::class_<std::shared_ptr<ouster::sensor_utils::record_handle>>(m, "record_handle")
        .def("__init__", [](std::shared_ptr<ouster::sensor_utils::record_handle>& self) {
             new (&self) std::shared_ptr<ouster::sensor_utils::record_handle>{
                 ouster::sensor_utils::record_handle_init()};
        });

    m.def("replay_initialize", &ouster::sensor_utils::replay_initialize);
    
    m.def("replay_uninitialize", [](std::shared_ptr<ouster::sensor_utils::playback_handle>& handle) {
        ouster::sensor_utils::replay_uninitialize(*handle);
    });
    m.def("replay_reset", [](std::shared_ptr<ouster::sensor_utils::playback_handle>& handle) {
        ouster::sensor_utils::replay_reset(*handle);
    });
    
    m.def("next_packet_info", [](std::shared_ptr<ouster::sensor_utils::playback_handle>& handle,
                                 ouster::sensor_utils::packet_info& packet_info) -> bool {
        return ouster::sensor_utils::next_packet_info(*handle, packet_info);
    });
    
    m.def("read_packet",
          [](std::shared_ptr<ouster::sensor_utils::playback_handle>& handle, py::buffer buf) -> size_t {
              return ouster::sensor_utils::read_packet(*handle, getptr(buf), getptrsize(buf));
          });
    
    m.def("record_initialize", &ouster::sensor_utils::record_initialize,
          py::arg("file_name"), py::arg("src_ip"), py::arg("dst_ip"),
          py::arg("frag_size"), py::arg("use_sll_encapsulation") = false);
    m.def("record_uninitialize",
          [](std::shared_ptr<ouster::sensor_utils::record_handle>& handle) {
              ouster::sensor_utils::record_uninitialize(*handle);
          });
    
    m.def("record_packet",
          [](std::shared_ptr<ouster::sensor_utils::record_handle>& handle,
             int src_port, int dst_port, py::buffer buf, double timestamp) {
              ouster::sensor_utils::record_packet(*handle,
                                                  src_port,
                                                  dst_port,
                                                  getptr(buf),
                                                  getptrsize(buf),
                                                  llround(timestamp * 1e6));

          });
    // clang-format on
    return m.ptr();
}