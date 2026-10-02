#pragma once
#include "../BoxGeometry.hpp"
#include <memory>
namespace System {
struct System { std::shared_ptr<BoxGeometry> box_geo = std::make_shared<BoxGeometry>(); };
inline System &get_system() { static System s; return s; }
}
