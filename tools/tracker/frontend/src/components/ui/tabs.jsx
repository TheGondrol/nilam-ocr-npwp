import * as React from "react"
import { cn } from "@/lib/utils"

const TabsContext = React.createContext({ active: '', onChange: () => {} })

function Tabs({ value, onValueChange, defaultValue, className, children, ...props }) {
  const [internalValue, setInternalValue] = React.useState(defaultValue || '')
  const current = value !== undefined ? value : internalValue
  const handleSelect = (v) => {
    if (onValueChange) onValueChange(v)
    else setInternalValue(v)
  }

  return (
    <TabsContext.Provider value={{ active: current, onChange: handleSelect }}>
      <div className={cn("w-full", className)} {...props}>
        {children}
      </div>
    </TabsContext.Provider>
  )
}

function TabsList({ className, ...props }) {
  return (
    <div
      className={cn(
        "inline-flex h-9 items-center justify-center rounded-lg bg-muted/60 p-1 text-muted-foreground border border-border/50",
        className
      )}
      {...props}
    />
  )
}

function TabsTrigger({ value, className, children, ...props }) {
  const { active, onChange } = React.useContext(TabsContext)
  const isSelected = active === value

  return (
    <button
      type="button"
      role="tab"
      aria-selected={isSelected}
      onClick={() => onChange(value)}
      className={cn(
        "inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1 text-xs font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 cursor-pointer select-none",
        isSelected
          ? "bg-background text-foreground shadow-sm font-semibold border border-border/60"
          : "hover:bg-background/40 hover:text-foreground text-muted-foreground",
        className
      )}
      {...props}
    >
      {children}
    </button>
  )
}

function TabsContent({ value, className, children, ...props }) {
  const { active } = React.useContext(TabsContext)
  if (active !== value) return null

  return (
    <div
      role="tabpanel"
      className={cn("mt-4 ring-offset-background focus-visible:outline-none", className)}
      {...props}
    >
      {children}
    </div>
  )
}

export { Tabs, TabsList, TabsTrigger, TabsContent }
