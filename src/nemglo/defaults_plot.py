import plotly.io as pio

PALETTEZZZ = ['#b2d4ee','#849db1','#4f6980',
           '#B4E3BC','#89AE8F','#638b66',
           '#ffb04f','#de9945','#af7635',
           '#ff7371','#d6635f','#b65551',
           '#AD134C','#cc688d','#ff82b0']

NORD = {
    'polar_1': '#4c566a',
    'polar_2': '#434c5e',
    'polar_3': '#3b4252',
    'polar_4': '#2e3440',
    'snow_1': '#eceff4',
    'snow_2': '#e5e9f0',
    'snow_3': '#d8dee9',
    'frost_1': '#5e81ac',
    'frost_2': '#81a1c1',
    'frost_3': '#88c0d0',
    'frost_4': '#8fbcbb',
    'aurora_p': '#b48ead',
    'aurora_g': '#a3be8c',
    'aurora_y': '#ebcb8b',
    'aurora_o': '#d08770',
    'aurora_r': '#bf616a',
}

PAL22 = {
    'blue_1': '#b2d4ee',
    'blue_2': '#849db1',
    'blue_3': '#4f6980',
    'green_1': '#B4E3BC',
    'green_2': '#89AE8F',
    'green_3': '#638b66',
    'green_4': '#445F46',
    'gold_1': '#ffb04f',
    'gold_2': '#de9945',
    'gold_3': '#af7635',
    'gold_3b': '#e2c29d',
    'salmon_1': '#ff7371',
    'salmon_2': '#d6635f',
    'salmon_3': '#b65551',
    'purple_1': '#CA4A72',
    'purple_2': '#9D496D',
    'purple_3': '#772D4D',
    'crimson_1': '#AD134C',
    'lime_1': '#D4DD7F',
    'lime_2': '#9DAF7D',
    'lime_3': '#808D69',
    'brown_1': '#E4B49C',
    'brown_2': '#BF936F',
    'brown_3': '#9F7553',
    'teal_1': '#7FE1E1',
    'teal_2': '#5BC8C5',
    'teal_3': '#4E9A9C'
}

NEU5 = {
    'tan': '#CEA07E',
    'purple': '#8F6593',
    'blue': '#4F759B',
    'green': '#638b66',
    'red': '#B9314F'
}

NEU5L = {
    'black': '#676765',
    'tan': '#C7916B',
    'purple': '#A179A4',
    'blue': '#5C83AD',
    'green': '#709973',
    'red': '#D86E85'  
}

# Plotly Sheet Theme
def NORD_theme():
    plotly_NORD_theme = pio.templates["simple_white"]
    plotly_NORD_theme.layout.plot_bgcolor = "#f4f4f5" 
    plotly_NORD_theme.layout.paper_bgcolor = "#FFFFFF"
    plotly_NORD_theme.layout.xaxis.gridcolor = NORD['snow_3']
    plotly_NORD_theme.layout.yaxis.gridcolor = NORD['snow_3']
    return plotly_NORD_theme

def NORD_theme_dark():
    plotly_NORD_theme = pio.templates["plotly_dark"]
    plotly_NORD_theme.layout.plot_bgcolor = "rgb(25,25,25)"
    #plotly_NORD_theme.layout.paper_bgcolor = "#FFFFFF"
    plotly_NORD_theme.layout.xaxis.gridcolor = NORD['polar_3']
    plotly_NORD_theme.layout.yaxis.gridcolor = NORD['polar_3']
    return plotly_NORD_theme


def set_font_size(layout, font_size=16):
    layout['titlefont']['size'] = font_size + 4
    layout.legend['font']['size'] = font_size

    for ax in [item for item in layout if item.__contains__('xaxis')]:
        layout[ax].titlefont.size = font_size
        layout[ax].tickfont.size = font_size

    for ax in [item for item in layout if item.__contains__('yaxis')]:
        layout[ax].titlefont.size = font_size
        layout[ax].tickfont.size = font_size


class plt_size():
    def slim():
        layout = dict(
            width=1000,
            height=250
        )
        return layout
    def small():
        layout = dict(
            width=1000,
            height=450
        )
        return layout
    def standard():
        layout = dict(
            width=1000,
            height=600
        )
        return layout
    def medium():
        layout = dict(
            width=1000,
            height=800
        )
        return layout
    def tall():
        layout = dict(
            width=1000,
            height=1000
        )
        return layout

class plt_markup():
    def legend_bottom():
        layout = dict(
            legend=dict(
                xanchor='center',
                x=0.5,
                y=-0.1,
                orientation='h', 
            )
        )
        return layout

    def report_margins():
        layout = dict(
            title=None,
            margin=dict(l=20, r=20, t=20, b=0),
        )
        return layout

class plt_save():
    def svg(fig, filename):
        width_in_mm, width_default_px, dpi = 160, 1000, 800
        scale = (width_in_mm / 25.4) / (width_default_px / dpi)
        fig.write_image(f'{filename}.svg',scale=scale)
        return