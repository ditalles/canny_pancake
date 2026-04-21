// Grape-scale test plate — compare print orientations
// All body + roof variants at 0.3 scale on one build plate
//
// ROW 1 (back):  Body variants
//   A: flipped (current, no supports)
//   B: upright (shows tier overhangs)
//   C+D: split halves (left/right at Y=0)
//
// ROW 2 (front): Roof variants
//   A: normal (barrel vault overhangs)
//   B: upside-down (ridge on bed)
//   C+D: split halves (front/back at Y=0)
//
// USAGE:
//   Plate: Render as-is → one STL with all pieces
//   Individual: openscad -D 'variant="body_flipped"' -o file.stl ...
//
// variant options: "plate", "body_flipped", "body_upright",
//   "body_split_F", "body_split_R", "roof_normal",
//   "roof_flipped", "roof_split_F", "roof_split_B"

use <lantern_model.scad>

$fn = 60;
sc = 0.3;
variant = "plate";

// Full-scale reference heights
body_ht = 76.5;
roof_ht = 36;

if (variant == "plate") {

    // === Body variants (back row) ===

    // A: Flipped — largest tier on bed (current, support-free)
    translate([0, 15, 0])
        scale([sc, sc, sc]) body_print();

    // B: Upright — smallest tier on bed (tier overhangs need supports)
    translate([30, 15, 0])
        scale([sc, sc, sc]) body();

    // C: Split front half (Y > 0), flat face on bed
    translate([55, 20, 0])
        scale([sc, sc, sc])
            difference() {
                body_print();
                translate([0, -100, 0]) cube(200, center=true);
            }

    // D: Split back half (Y < 0), flat face on bed
    translate([55, 10, 0])
        scale([sc, sc, sc])
            difference() {
                body_print();
                translate([0, 100, 0]) cube(200, center=true);
            }

    // === Roof variants (front row) ===

    // A: Normal — base plate on bed (vault overhangs at top)
    translate([0, -20, 0])
        scale([sc, sc, sc]) roof_with_groove();

    // B: Upside-down — ridge on bed
    translate([30, -20, 0])
        scale([sc, sc, sc])
            translate([0, 0, roof_ht])
                mirror([0, 0, 1])
                    roof_with_groove();

    // C: Split front half (Y > 0)
    translate([55, -15, 0])
        scale([sc, sc, sc])
            difference() {
                roof_with_groove();
                translate([0, -100, 0]) cube(200, center=true);
            }

    // D: Split back half (Y < 0)
    translate([55, -25, 0])
        scale([sc, sc, sc])
            difference() {
                roof_with_groove();
                translate([0, 100, 0]) cube(200, center=true);
            }

} else if (variant == "body_flipped") {
    scale([sc, sc, sc]) body_print();

} else if (variant == "body_upright") {
    scale([sc, sc, sc]) body();

} else if (variant == "body_split_F") {
    scale([sc, sc, sc])
        difference() {
            body_print();
            translate([0, -100, 0]) cube(200, center=true);
        }

} else if (variant == "body_split_B") {
    scale([sc, sc, sc])
        difference() {
            body_print();
            translate([0, 100, 0]) cube(200, center=true);
        }

} else if (variant == "roof_normal") {
    scale([sc, sc, sc]) roof_with_groove();

} else if (variant == "roof_flipped") {
    scale([sc, sc, sc])
        translate([0, 0, roof_ht])
            mirror([0, 0, 1])
                roof_with_groove();

} else if (variant == "roof_split_F") {
    scale([sc, sc, sc])
        difference() {
            roof_with_groove();
            translate([0, -100, 0]) cube(200, center=true);
        }

} else if (variant == "roof_split_B") {
    scale([sc, sc, sc])
        difference() {
            roof_with_groove();
            translate([0, 100, 0]) cube(200, center=true);
        }
}
