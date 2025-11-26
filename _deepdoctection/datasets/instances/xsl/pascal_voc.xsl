<?xml version="1.0" encoding="UTF-8"?>
<!--Stylesheet parameters: NONE-->

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:template match="/">
        <xsl:text>{"annotation":"</xsl:text>
        <xsl:value-of select="annotation/folder"/>
        <xsl:text>", "filename":"</xsl:text>
        <xsl:value-of select="annotation/filename"/>
        <xsl:text>", "path":"</xsl:text>
        <xsl:value-of select="annotation/path"/>
        <xsl:text>", "database":"</xsl:text>
        <xsl:value-of select="annotation/source/database"/>
        <xsl:text>", "width":"</xsl:text>
        <xsl:value-of select="annotation/size/width"/>
        <xsl:text>", "height":"</xsl:text>
        <xsl:value-of select="annotation/size/height"/>
        <xsl:text>", "depth":"</xsl:text>
        <xsl:value-of select="annotation/size/depth"/>
        <xsl:text>", "segmented":"</xsl:text>
        <xsl:value-of select="annotation/segmented"/>
        <xsl:text>", "objects": [</xsl:text>
        <xsl:for-each select="annotation/object">
            <xsl:text>{"name":"</xsl:text>
            <xsl:value-of select="name"/>
            <xsl:text>","pose":"</xsl:text>
            <xsl:value-of select="pose"/>
            <xsl:text>","truncated":"</xsl:text>
            <xsl:value-of select="truncated"/>
            <xsl:text>","difficult":</xsl:text>
            <xsl:value-of select="difficult"/>
            <xsl:text>,"xmin":</xsl:text>
            <xsl:value-of select="bndbox/xmin"/>
            <xsl:text>,"ymin":</xsl:text>
            <xsl:value-of select="bndbox/ymin"/>
            <xsl:text>,"xmax":</xsl:text>
            <xsl:value-of select="bndbox/xmax"/>
            <xsl:text>,"ymax":</xsl:text>
            <xsl:value-of select="bndbox/ymax"/>
            <xsl:text>},</xsl:text>
        </xsl:for-each>
        <xsl:text>]</xsl:text>
        <xsl:text>&#xa; }</xsl:text>
    </xsl:template>
</xsl:stylesheet>