"""
    GroupHorseshoeFamily(groups; scale=nothing, starts=nothing)

Define one group label per coefficient. Independent unit half-Cauchy global, group,
and local scales give conditional coefficient standard deviations
`scale * global_scale * group_scales[group] * local_scales`.
`scale` fixes the global scale multiplier; `nothing` fits `log_scale`.
The joint measure retains `coefficients`, `global_scale`, `group_scales`, and
`local_scales`. Group scales follow label order of first appearance.

This is the one-level grouped global-local horseshoe of Xu et al. (2017),
[Section 2.3](https://arxiv.org/abs/1709.04333). A singleton group still has both
a group scale and a local scale.
"""
struct GroupHorseshoeFamily{G,I,S,T} <: AbstractCompoundFamily
    groups::G
    indices::I
    scale::S
    starts::T
end
function GroupHorseshoeFamily(groups; scale=nothing, starts=nothing)
    labels = Tuple(groups)
    isempty(labels) &&
        throw(ArgumentError("group labels must contain one entry per coefficient"))
    _compound_positive(scale, :scale)
    names = unique(labels)
    indices = map(label -> something(findfirst(isequal(label), names)), labels)
    GroupHorseshoeFamily(labels, indices, scale, starts)
end
_compound_shape(family::GroupHorseshoeFamily) = (length(family.groups),)
function _compound_starts(family::GroupHorseshoeFamily)
    if family.starts isa NamedTuple
        return (family.starts,)
    elseif family.starts !== nothing
        return family.starts
    elseif family.scale === nothing
        return ((log_scale=0.0,),)
    end
    (NamedTuple(),)
end
