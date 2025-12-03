# -*- coding: utf-8 -*-
"""通用 UI 组件与封装，供主界面及子页面复用。"""

import streamlit as st


def render_table(df, *, key=None, hide_index=True, editable=False):
    """
    统一渲染表格(对旧/新 Streamlit 都安全):
    - editable=False: 只读(用 data_editor disabled=True 以保留 key)
    - editable=True : 可编辑
    - 不再传 width 参数.避免 'str' as int 的报错
    """
    try:
        if editable:
            return st.data_editor(
                df,
                hide_index=hide_index,
                key=key,
            )
        if key is not None:
            return st.data_editor(
                df,
                hide_index=hide_index,
                disabled=True,
                key=key,
            )
        return st.dataframe(df, hide_index=hide_index)
    except TypeError:
        return st.data_editor(
            df,
            hide_index=hide_index,
            disabled=not editable,
            key=key,
        )

