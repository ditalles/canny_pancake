// Turkish/Ottoman Hanging Lantern — 4-sided, 3-tier, barrel-vault top
// Based on traditional Antalya-style street lantern
// Render: F5 (preview) or F6 (full render) in OpenSCAD

$fn = 40;

// ─── Parameters ───

frame_t = 3;          // Metal frame thickness
glass_t = 1;          // Glass panel thickness

// Tier dimensions [width, depth, height] — largest on top, smallest at bottom
tier1_w = 70;         // Top tier width
tier1_d = 50;         // Top tier depth
tier1_h = 50;         // Top tier height

tier2_w = 55;         // Middle tier width
tier2_d = 40;         // Middle tier depth
tier2_h = 45;         // Middle tier height

tier3_w = 35;         // Bottom tier width
tier3_d = 28;         // Bottom tier depth
tier3_h = 25;         // Bottom tier height

band_h = 5;           // Height of frame band between tiers
roof_h = 20;          // Barrel vault roof height

// Bracket
bracket_arm = 100;    // Horizontal reach
bracket_rod = 4;      // Rod diameter
chain_len = 30;       // Chain length

// ─── Modules ───

// Solid box frame (outer shell minus inner cavity)
module box_frame(w, d, h) {
    difference() {
        cube([w, d, h], center=true);
        cube([w - frame_t*2, d - frame_t*2, h + 1], center=true);
    }
}

// Glass panel with rectangular grid pattern (mullions)
module glass_panel_with_grid(panel_w, panel_h, cols, rows) {
    // Base glass panel
    color([0.9, 0.9, 0.85, 0.6])
        cube([glass_t, panel_w, panel_h], center=true);

    // Horizontal mullions
    mullion_t = 2;
    for (r = [0:rows]) {
        y_pos = -panel_h/2 + r * (panel_h / rows);
        color([0.15, 0.12, 0.1])
            cube([mullion_t + 0.5, panel_w, mullion_t], center=true);
        if (r < rows) {
            // nothing — just boundary bars
        }
    }

    // Vertical mullions
    for (c = [0:cols]) {
        x_pos = -panel_w/2 + c * (panel_w / cols);
        color([0.15, 0.12, 0.1])
            translate([0, x_pos, 0])
                cube([mullion_t + 0.5, mullion_t, panel_h], center=true);
    }

    // Horizontal mullions (cross bars)
    for (r = [0:rows]) {
        z_pos = -panel_h/2 + r * (panel_h / rows);
        color([0.15, 0.12, 0.1])
            translate([0, 0, z_pos])
                cube([mullion_t + 0.5, panel_w, mullion_t], center=true);
    }
}

// A complete tier: frame + 4 glass panels with grid
module tier(w, d, h, cols, rows) {
    // Dark metal frame
    color([0.15, 0.12, 0.1])
        box_frame(w, d, h);

    // Top and bottom plates
    color([0.15, 0.12, 0.1]) {
        translate([0, 0, h/2 - frame_t/2])
            cube([w, d, frame_t], center=true);
        translate([0, 0, -h/2 + frame_t/2])
            cube([w, d, frame_t], center=true);
    }

    glass_w_front = w - frame_t * 3;
    glass_w_side = d - frame_t * 3;
    glass_h = h - frame_t * 3;

    // Front and back panels
    for (sign = [-1, 1]) {
        translate([0, sign * d/2, 0])
            rotate([0, 0, 0])
                glass_panel_with_grid(glass_w_front, glass_h, cols, rows);
    }

    // Left and right panels
    for (sign = [-1, 1]) {
        translate([sign * w/2, 0, 0])
            rotate([0, 0, 90])
                glass_panel_with_grid(glass_w_side, glass_h, cols, rows);
    }
}

// Band between tiers
module frame_band(w, d) {
    color([0.15, 0.12, 0.1])
        cube([w + 4, d + 4, band_h], center=true);
}

// Barrel vault roof (half-cylinder)
module barrel_roof(w, d) {
    color([0.15, 0.12, 0.1]) {
        // Half cylinder running along the width
        intersection() {
            translate([0, 0, 0])
                rotate([0, 90, 0])
                    cylinder(r=d/2 + 3, h=w + 6, center=true, $fn=40);
            translate([0, 0, roof_h/2])
                cube([w + 8, d + 8, roof_h + 2], center=true);
        }

        // Flat cap on the ends
        for (sign = [-1, 1]) {
            translate([sign * (w/2 + 1), 0, 0])
                intersection() {
                    rotate([0, 90, 0])
                        cylinder(r=d/2 + 2, h=3, center=true, $fn=40);
                    translate([0, 0, roof_h/2])
                        cube([5, d + 6, roof_h], center=true);
                }
        }

        // Ridge bar on top
        translate([0, 0, d/2 + 2])
            cube([w + 6, 3, 3], center=true);
    }

    // Glass end panels (arched)
    for (sign = [-1, 1]) {
        color([0.9, 0.9, 0.85, 0.4])
        translate([sign * (w/2), 0, 0])
            intersection() {
                rotate([0, 90, 0])
                    cylinder(r=d/2, h=glass_t, center=true, $fn=40);
                translate([0, 0, roof_h/2])
                    cube([glass_t + 1, d, roof_h], center=true);
            }
    }

    // Front/back glass on roof
    for (sign = [-1, 1]) {
        color([0.9, 0.9, 0.85, 0.4])
        translate([0, sign * (d/2), 0])
            intersection() {
                cube([w, glass_t, roof_h * 2], center=true);
                translate([0, 0, 0])
                    rotate([0, 90, 0])
                        cylinder(r=d/2, h=w, center=true, $fn=40);
                translate([0, 0, roof_h/2])
                    cube([w + 1, glass_t + 1, roof_h + 1], center=true);
            }
    }
}

// Chain link
module chain_link(h) {
    color([0.2, 0.18, 0.15]) {
        links = floor(h / 8);
        for (i = [0:links-1]) {
            translate([0, 0, -i * 8]) {
                if (i % 2 == 0) {
                    difference() {
                        cube([3, 6, 8], center=true);
                        cube([1.5, 4, 6], center=true);
                    }
                } else {
                    difference() {
                        cube([6, 3, 8], center=true);
                        cube([4, 1.5, 6], center=true);
                    }
                }
            }
        }
    }
}

// Wall bracket with scroll
module wall_bracket() {
    color([0.15, 0.12, 0.1]) {
        // Wall plate
        translate([-bracket_arm, 0, 0])
            cube([8, 30, 50], center=true);

        // Horizontal arm
        translate([-bracket_arm/2, 0, 20])
            rotate([0, 5, 0])
                cube([bracket_arm, bracket_rod, bracket_rod], center=true);

        // Diagonal support
        translate([-bracket_arm * 0.7, 0, 10])
            rotate([0, 35, 0])
                cube([bracket_arm * 0.5, bracket_rod, bracket_rod], center=true);

        // Decorative scroll (circle at the end)
        translate([-bracket_arm * 0.3, 0, 15]) {
            difference() {
                cylinder(r=12, h=bracket_rod, center=true, $fn=30);
                cylinder(r=9, h=bracket_rod + 1, center=true, $fn=30);
            }
        }

        // Hook at end for chain
        translate([5, 0, 22])
            difference() {
                cylinder(r=6, h=bracket_rod, center=true, $fn=20);
                cylinder(r=4, h=bracket_rod + 1, center=true, $fn=20);
                translate([0, -6, 0])
                    cube([14, 6, bracket_rod + 2], center=true);
            }
    }
}

// ─── Full Assembly ───

module lantern() {
    // Tier positions (stacked bottom to top)
    z_tier3 = 0;
    z_band2 = z_tier3 + tier3_h/2 + band_h/2;
    z_tier2 = z_band2 + band_h/2 + tier2_h/2;
    z_band1 = z_tier2 + tier2_h/2 + band_h/2;
    z_tier1 = z_band1 + band_h/2 + tier1_h/2;
    z_roof  = z_tier1 + tier1_h/2;

    // Bottom tier — 2 panes, 1 row
    translate([0, 0, z_tier3])
        tier(tier3_w, tier3_d, tier3_h, 2, 1);

    // Band between bottom and middle
    translate([0, 0, z_band2])
        frame_band(tier3_w, tier3_d);

    // Middle tier — 2 panes, 2 rows
    translate([0, 0, z_tier2])
        tier(tier2_w, tier2_d, tier2_h, 2, 2);

    // Band between middle and top
    translate([0, 0, z_band1])
        frame_band(tier2_w, tier2_d);

    // Top tier — 3 panes, 2 rows
    translate([0, 0, z_tier1])
        tier(tier1_w, tier1_d, tier1_h, 3, 2);

    // Barrel vault roof
    translate([0, 0, z_roof])
        barrel_roof(tier1_w, tier1_d);

    // Chain
    translate([0, 0, z_roof + roof_h])
        chain_link(chain_len);

    // Wall bracket
    translate([0, 0, z_roof + roof_h + chain_len])
        wall_bracket();

    // Bottom cap
    color([0.15, 0.12, 0.1])
    translate([0, 0, z_tier3 - tier3_h/2 - 3])
        cube([tier3_w * 0.7, tier3_d * 0.7, 4], center=true);
}

// ─── Render ───

lantern();
