import * as React from "react"
import { cva } from "class-variance-authority"
import { cn } from "@/lib/utils"

const alertVariants = cva(
  "relative w-full rounded-lg border px-4 py-3 text-xs [&>svg+div]:translate-y-[-3px] [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-3.5 [&>svg]:text-foreground [&>svg~*]:pl-7",
  {
    variants: {
      variant: {
        default: "bg-background text-foreground border-border",
        destructive:
          "border-destructive/30 text-destructive bg-destructive/10 dark:text-destructive-foreground [&>svg]:text-destructive",
        warning:
          "border-amber-500/30 text-amber-800 bg-amber-500/10 dark:text-amber-300 [&>svg]:text-amber-500",
        success:
          "border-emerald-500/30 text-emerald-800 bg-emerald-500/10 dark:text-emerald-300 [&>svg]:text-emerald-500",
        info:
          "border-sky-500/30 text-sky-800 bg-sky-500/10 dark:text-sky-300 [&>svg]:text-sky-500",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

function Alert({ className, variant, ...props }) {
  return (
    <div
      role="alert"
      className={cn(alertVariants({ variant }), className)}
      {...props}
    />
  )
}

function AlertTitle({ className, ...props }) {
  return (
    <h5
      className={cn("mb-1 font-semibold leading-none tracking-tight text-foreground", className)}
      {...props}
    />
  )
}

function AlertDescription({ className, ...props }) {
  return (
    <div
      className={cn("text-xs [&_p]:leading-relaxed text-muted-foreground", className)}
      {...props}
    />
  )
}

export { Alert, AlertTitle, AlertDescription }
