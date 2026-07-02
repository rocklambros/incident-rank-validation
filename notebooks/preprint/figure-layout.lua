-- figure-layout.lua — control preprint figure placement.
-- pandoc >= 3 wraps an attributed image in a Figure node; width/wrap live on
-- the inner Image, caption on the Figure. Emit wrapfigure or figure[htbp].

local function pct_to_frac(s)
  if not s then return 0.85 end
  local n = s:match("^(%d+%.?%d*)%%$")
  if n then return tonumber(n) / 100.0 end
  return tonumber(s) or 0.85
end

local function find_image(blocks)
  local found = nil
  for _, b in ipairs(blocks) do
    if b.content then
      for _, inl in ipairs(b.content) do
        if inl.t == "Image" then found = inl; break end
      end
    end
    if found then break end
  end
  return found
end

local function caption_latex(caption)
  if not caption or not caption.long then return "" end
  local s = pandoc.write(pandoc.Pandoc(caption.long), "latex")
  return (s:gsub("%s+$", ""))
end

function Figure(fig)
  local img = find_image(fig.content)
  if not img then return nil end
  local w = pct_to_frac(img.attributes.width)
  local wrap = img.attributes.wrap
  local src = img.src
  local cap = caption_latex(fig.caption)
  local latex
  if wrap == "left" or wrap == "right" then
    local side = (wrap == "left") and "L" or "R"
    latex = string.format(
      "\\begin{wrapfigure}{%s}{%.2f\\textwidth}\n\\centering\n" ..
      "\\includegraphics[width=%.2f\\textwidth]{%s}\n\\caption{%s}\n\\end{wrapfigure}",
      side, w + 0.03, w, src, cap)
  else
    if wrap ~= nil then
      io.stderr:write("figure-layout.lua: unknown wrap='" .. tostring(wrap) ..
                      "' for " .. src .. "; centering\n")
    end
    latex = string.format(
      "\\begin{figure}[htbp]\n\\centering\n" ..
      "\\includegraphics[width=%.2f\\textwidth]{%s}\n\\caption{%s}\n\\end{figure}",
      w, src, cap)
  end
  return pandoc.RawBlock("latex", latex)
end
