# SPDX-License-Identifier: GPL-3.0-or-later
"""Plain-English result text for the panel. Built in the worker, drawn by the add-on as it is."""

MAP_NAMES = dict(basecolor="Base Colour", roughness="Roughness", metallic="Metallic", emission="Emission",
                 alpha="Alpha", ao="Ambient Occlusion")
LARGE_INPUT_TRIS = 5000000
REFUSE_INPUT_TRIS = 8000000

RIGGED = ("The source is rigged. The asset is built from the rest shape: the armature, the weights and the "
          "animation are not carried over.")


def number(n):
    return "{:,}".format(int(n))


def percent(x):
    return ("%.2f" % x).rstrip("0").rstrip(".") + " %"


def whole_percent(share):
    return "%d %%" % round(100 * share)


def quad_lines(low):
    """Result lines of a quad-dominant asset: face count, quad / triangle / thin shares."""
    quad = low["quad"]
    return [
        "%s triangles -> %s faces" % (number(low["tris_in"]), number(quad["faces"])),
        "Quads %s, triangles %s of the faces" % (whole_percent(quad["quad_share"]), whole_percent(quad["tri_share"])),
        "Thin parts (kept as triangles): %s of the surface" % whole_percent(quad["thin_share"]),
        "In a game engine: %s triangles" % number(low["tris"]),
    ]


def quad_messages(low, mode):
    """Messages about the Quad-dominant mode: the fallback and its reason, or what the triangles are."""
    quad = low.get("quad")
    if mode != "QUADS" or not quad:
        return []
    out = []
    if not quad["used"]:
        out.append(["WARNING", "Quad-dominant was not possible here, a Game asset (triangles) was made instead. "
                               "Reason: %s." % quad["reason"]])
        if quad.get("quad_share") is not None:
            out.append(["INFO", "The quad mesh that was set aside: quads %s, triangles %s of the faces, thin parts "
                                "%s of the surface." % (whole_percent(quad["quad_share"]),
                                                        whole_percent(quad["tri_share"]),
                                                        whole_percent(quad["thin_share"]))])
        elif quad.get("thin_share") is not None:
            out.append(["INFO", "Thin parts: %s of the surface." % whole_percent(quad["thin_share"])])
        return out
    if quad["tri_share"] > 0.02:
        out.append(["INFO", "Thin parts (thinner than about one quad) and the seams next to them are triangles; "
                            "everything else is quads. A higher budget gives a higher quad share."])
    if low["budget"] < 10000:
        out.append(["INFO", "Below 10,000 faces the quad share drops. The mode was tuned at 10,000."])
    if low.get("short"):
        runs = max(1, int(quad.get("target_runs") or low.get("passes") or 1))
        out.append(["WARNING", "The budget is %s faces, the asset has %s: the remesher did not get closer in "
                               "%d pass%s." % (number(low["budget"]), number(quad["faces"]), runs, "" if runs == 1 else "es")])
    out.append(["INFO", "The textures were baked on a fixed split of every quad; Export GLB writes exactly those "
                        "triangles."])
    return out


def asset_report(name, low, dev, uv, texels, maps, size, seconds, mode="GAME"):
    """One asset: (lines, messages). messages are [level, text] with level INFO or WARNING."""
    quads_kept = bool(low.get("quad") and low["quad"]["used"])
    lines = (quad_lines(low) if quads_kept else ["%s -> %s triangles" % (number(low["tris_in"]), number(low["tris"]))]) + [
        "Shape: within %s of the size (largest gap %s)" % (percent(dev["p99"]), percent(dev["max"])),
        "Texture %d px: Normal, %s" % (size, ", ".join(MAP_NAMES[m] for m in maps)) if maps
        else "Texture %d px: Normal" % size,
        "%d UV islands, %s vertices in a game engine" % (uv["islands"], number(uv["engine_verts"])),
        "Time: %.0f s" % seconds,
    ]
    messages = quad_messages(low, mode)
    if low["under_budget"]:
        messages.append(["INFO", "The source already has fewer triangles than the budget: the geometry is kept "
                                 "as it is."])
    if low["hidden_removed_share"] > 0 and quads_kept:
        messages.append(["INFO", "%s of the surface is inner layers that cannot be seen from outside (the file has a "
                                 "second skin under the visible one). They were left out and the openings closed, so "
                                 "the quads follow the visible surface. With \"Remove hidden surface\" unticked "
                                 "such a file becomes a Game asset." % percent(100 * low["hidden_removed_share"])])
    elif low["hidden_removed_share"] > 0:
        messages.append(["INFO", "%s of the surface is enclosed by other surface (no test ray reaches it from "
                                 "outside) and was removed: the mesh is open there. Untick \"Remove hidden "
                                 "surface\" to keep it." % percent(100 * low["hidden_removed_share"])])
    if low.get("unlock_rounds") or low.get("merged_all"):
        messages.append(["INFO", "Parts of the source are tangled (faces that share one edge with several "
                                 "neighbours or touch at single points) and cannot be reduced the normal way. "
                                 "To keep the budget their points were merged by distance: these places can "
                                 "look rougher than the rest."])
    if quads_kept:
        pass                                       # the budget counts faces there: quad_messages says it
    elif low.get("short") == "over":
        messages.append(["WARNING", "The budget is %s triangles, the asset has %s. The mesh could not be reduced "
                                    "further: what is left are tangled or separate pieces that fall apart when "
                                    "they are merged more." % (number(low["budget"]), number(low["tris"]))])
    elif low.get("short") == "under":
        why = ("that is all the visible surface kept after the hidden surface was removed. Untick \"Remove hidden "
               "surface\" to get closer to the budget." if low["hidden_removed_share"] > 0
               else "the source has no more usable triangles after cleaning.")
        messages.append(["WARNING", "The budget is %s triangles, the asset has only %s: %s" % (
            number(low["budget"]), number(low["tris"]), why)])
    if dev["p99"] > 0.5:
        messages.append(["WARNING", "The low mesh leaves the source shape by %s of the size. Try a higher "
                                    "budget." % percent(dev["p99"])])
    if texels.get("empty"):
        messages.append(["WARNING", "The bake found no source surface: the textures are empty."])
    else:
        share = 100.0 * texels["missed"] / max(1, texels["texels"])
        if texels["missed"]:
            text = "Missed texels: %s of %s (filled from their neighbours)." % (
                number(texels["missed"]), number(texels["texels"]))
            messages.append(["WARNING" if share > 0.5 else "INFO", text])
        back = 100.0 * texels.get("back_normal", 0) / max(1, texels["texels"])
        if back > 2.0:
            messages.append(["WARNING", "%s of the texture was baked from the back of a surface. This happens "
                                        "on one-sided sheets and where parts almost touch." % percent(back)])
    if uv["zero_area_tris"]:
        messages.append(["INFO", "%d very small triangles got no texture area." % uv["zero_area_tris"]])
    return lines, messages


def job_messages(notes, part_notes, total_tris, replaced_materials, missing_images, alpha_kept_hidden,
                 alpha_baked=False, vertex_colours=False, unlit=()):
    """Messages about the whole job (source and scope)."""
    out = []
    if notes["rigged"] or notes["animated"]:
        out.append(["WARNING", RIGGED])
    if notes["shape_keys"]:
        out.append(["WARNING", "The source has shape keys: the basis shape is used, the shape keys are dropped."])
    if notes["modifiers_applied"]:
        out.append(["INFO", "Modifiers of the source were applied for the asset (the source is not changed)."])
    if total_tris > LARGE_INPUT_TRIS:
        out.append(["WARNING", "Large input: %s triangles. This takes several minutes and several GB of "
                               "memory." % number(total_tris)])
    if replaced_materials:
        out.append(["WARNING", "%d material(s) have no Principled BSDF or Emission shader on the surface (a "
                               "shader inside a node group is not read) and were baked as plain grey: %s." % (
            len(replaced_materials), ", ".join(replaced_materials[:4]))])
    if vertex_colours:
        out.append(["INFO", "Faces without a readable material were baked from the mesh's vertex colours."])
    if unlit:
        out.append(["INFO", "Unlit material(s) (Emission shader): the emitted colour was baked as the base "
                            "colour: %s." % ", ".join(list(unlit)[:4])])
    if missing_images:
        out.append(["WARNING", "Texture file not found: %s." % ", ".join(missing_images[:4])])
    if alpha_baked:
        out.append(["INFO", "Transparency is baked as a cut-out: half-transparent surface such as glass becomes "
                            "solid or a hole."])
    if alpha_kept_hidden:
        out.append(["INFO", "The source has transparent parts, so surface behind them was kept."])
    refused = part_notes.get("refused")
    if refused == "parts":
        out.append(["WARNING", "The selection has too many separate parts for one asset each: one asset was "
                               "built for the whole selection."])
    elif refused == "pieces":
        out.append(["WARNING", "The selection falls into more than 2,000 loose pieces: one asset was built for "
                               "the whole selection."])
    if part_notes.get("small_pieces_merged"):
        out.append(["INFO", "%d small loose piece(s) were kept with the nearest large part." %
                    part_notes["small_pieces_merged"]])
    return out
