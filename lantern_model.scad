// Traditional Hexagonal Wall Lantern
// Render in OpenSCAD: F5 (preview) or F6 (full render)
// Export as STL for 3D printing

$fn = 6;  // Hexagonal cross-section

// ─── Parameters ───
lantern_scale = 1;  // Scale factor (1 = roughly 300mm tall)

// Body dimensions
hex_radius = 40;        // Outer radius of hexagonal body
wall_thickness = 3;
glass_inset = 1.5;

// Section heights
cap_height = 15;
upper_frame_height = 8;
upper_glass_height = 55;
mid_band_height = 8;
lower_glass_height = 45;
lower_frame_height = 8;
base_height = 12;
finial_height = 20;

// Bracket dimensions
bracket_length = 120;
bracket_width = 12;
bracket_thickness = 6;
bracket_curve_radius = 60;

// Wall plate
plate_width = 50;
plate_height = 80;
plate_depth = 8;

// ─── Modules ───

module hexagonal_prism(radius, height, center=true) {
    cylinder(r=radius, h=height, center=center, $fn=6);
}

module hexagonal_tube(outer_r, inner_r, height) {
    difference() {
        hexagonal_prism(outer_r, height, center=false);
        translate([0, 0, -0.1])
            hexagonal_prism(inner_r, height + 0.2, center=false);
    }
}

// Roof cap — pyramid top
module roof_cap() {
    // Pyramid
    cylinder(r1=hex_radius + 5, r2=3, h=cap_height + 15, $fn=6);
    // Lip overhang
    translate([0, 0, -3])
        hexagonal_prism(hex_radius + 8, 3, center=false);
}

// Decorative finial on top
module finial() {
    // Spike
    cylinder(r1=4, r2=1.5, h=finial_height, $fn=12);
    // Ball
    translate([0, 0, 5])
        sphere(r=4, $fn=16);
}

// Frame band — solid hexagonal ring
module frame_band(height, radius=hex_radius) {
    hexagonal_tube(radius + 1, radius - wall_thickness, height);
}

// Glass section — hexagonal tube with cutouts for glass panels
module glass_section(height, radius=hex_radius) {
    difference() {
        hexagonal_tube(radius, radius - wall_thickness, height);

        // Cut glass panel openings on each face
        for (i = [0:5]) {
            rotate([0, 0, i * 60 + 30])
                translate([radius - wall_thickness/2, 0, height/2])
                    cube([wall_thickness + 2,
                          radius * 0.75,
                          height - 10],
                         center=true);
        }
    }

    // Thin glass panels (translucent)
    %for (i = [0:5]) {
        rotate([0, 0, i * 60 + 30])
            translate([radius - glass_inset, 0, height/2])
                cube([0.8,
                      radius * 0.72,
                      height - 12],
                     center=true);
    }
}

// Base plate under the lantern body
module base_plate() {
    // Tapered base
    cylinder(r1=hex_radius * 0.6, r2=hex_radius + 2, h=base_height, $fn=6);
    // Bottom cap
    translate([0, 0, -5])
        cylinder(r1=4, r2=hex_radius * 0.5, h=5, $fn=6);
    // Bottom finial
    translate([0, 0, -15])
        cylinder(r1=2, r2=4, h=10, $fn=12);
}

// Decorative scroll bracket
module bracket() {
    // Main horizontal arm
    translate([0, 0, 0])
    rotate([0, 90, 0]) {
        // Curved arm using hull between spheres
        for (t = [0:5:85]) {
            hull() {
                angle1 = t;
                angle2 = t + 5;
                translate([bracket_curve_radius * sin(angle1),
                           0,
                           bracket_curve_radius * (1 - cos(angle1))])
                    sphere(r=bracket_thickness/2, $fn=8);
                translate([bracket_curve_radius * sin(angle2),
                           0,
                           bracket_curve_radius * (1 - cos(angle2))])
                    sphere(r=bracket_thickness/2, $fn=8);
            }
        }
    }

    // Decorative scroll at the bottom
    translate([bracket_curve_radius - 10, 0, -15])
    rotate([0, 90, 0]) {
        for (t = [0:5:180]) {
            scroll_r = 15;
            hull() {
                translate([scroll_r * sin(t), 0, scroll_r * cos(t)])
                    sphere(r=2.5, $fn=8);
                translate([scroll_r * sin(t+5), 0, scroll_r * cos(t+5)])
                    sphere(r=2.5, $fn=8);
            }
        }
    }

    // Upper support strut
    translate([10, 0, 30])
    rotate([0, -35, 0])
        cylinder(r=bracket_thickness/2.5, h=70, $fn=8);
}

// Wall mounting plate
module wall_plate() {
    difference() {
        // Main plate
        cube([plate_depth, plate_width, plate_height], center=true);
        // Mounting holes
        for (dy = [-15, 15]) {
            for (dz = [-25, 25]) {
                translate([0, dy, dz])
                    rotate([0, 90, 0])
                        cylinder(r=3, h=plate_depth + 2, center=true, $fn=12);
            }
        }
    }
}

// ─── Assembly ───

module lantern_body() {
    z = 0;

    // Base
    translate([0, 0, z])
        base_plate();

    // Lower frame band
    translate([0, 0, z + base_height])
        frame_band(lower_frame_height);

    // Lower glass section
    translate([0, 0, z + base_height + lower_frame_height])
        glass_section(lower_glass_height);

    // Mid band
    translate([0, 0, z + base_height + lower_frame_height + lower_glass_height])
        frame_band(mid_band_height, hex_radius + 2);

    // Upper glass section
    translate([0, 0, z + base_height + lower_frame_height + lower_glass_height + mid_band_height])
        glass_section(upper_glass_height, hex_radius);

    // Upper frame band
    translate([0, 0, z + base_height + lower_frame_height + lower_glass_height +
               mid_band_height + upper_glass_height])
        frame_band(upper_frame_height);

    // Roof cap
    total_body = base_height + lower_frame_height + lower_glass_height +
                 mid_band_height + upper_glass_height + upper_frame_height;
    translate([0, 0, z + total_body])
        roof_cap();

    // Top finial
    translate([0, 0, z + total_body + cap_height + 15])
        finial();
}

module full_lantern() {
    total_body_height = base_height + lower_frame_height + lower_glass_height +
                        mid_band_height + upper_glass_height + upper_frame_height;
    lantern_center_z = total_body_height / 2 + base_height;

    // Wall plate
    translate([-bracket_curve_radius - plate_depth/2 + 5, 0, lantern_center_z + 20])
        wall_plate();

    // Bracket arm
    translate([-bracket_curve_radius + 10, 0, lantern_center_z + 50])
        bracket();

    // Lantern body
    lantern_body();
}

// ─── Render ───

scale([lantern_scale, lantern_scale, lantern_scale])
    full_lantern();
